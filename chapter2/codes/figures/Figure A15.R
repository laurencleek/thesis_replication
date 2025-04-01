# Data loading and processing
library(arrow)
library(tidyverse)
library(slider)
library(lubridate)
library(readxl)
library(patchwork)
library(ggpattern)
library(ggsankey)

# Map distance calculations
library(geosphere)
library(sf)
library(rnaturalearth)

# Tables
library(knitr)
library(kableExtra)

source("./codes/functions/chart_themes.R")
source("./codes/functions/diff_in_diff_functions.R")
source("./codes/constants/diff_in_diff_settings.R")

source("./codes/constants/varnames.R")

# Set bounds on CBI datasets
lower_bound <- (1997 - EVENT_STUDY_NUMBER_LAGS) # Need to align this with the (max) number of leads and lags and possibly if observations are dropped
upper_bound <- 2023 #

speech_aggregation <- read_parquet("./data/processed/aggregations/speech_agg.parquet") %>%
  filter(
    year >= 1997,
    year <= 2023
  ) %>%
  mutate(
    date = as.Date(date) # Date datatype is not preserved in parquet it seems
  ) %>%
  arrange(date)

speeches_countries <- setdiff(speech_aggregation %>% pull(country) %>% unique(), "")

f_v_period_mean <- speech_aggregation %>%
  filter(country == "USA") %>%
  select(year, date, monetary_dominance, fiscal_dominance) %>%
  mutate(
    fiscal_v_monetary = fiscal_dominance / (fiscal_dominance + monetary_dominance),
    fd_period = case_when(
      date < as.Date("2000-10-01") ~ "md_1",
      date > as.Date("2003-10-01") & date < as.Date("2004-07-01") ~ "md_2",
      date > as.Date("2005-10-01") & date < as.Date("2007-01-01") ~ "md_3",
      date > as.Date("2010-04-01") & date < as.Date("2011-01-01") ~ "md_4",
      date > as.Date("2015-10-01") ~ "md_5",
      date >= as.Date("2000-10-01") & date <= as.Date("2003-10-01") ~ "fd_1",
      date >= as.Date("2004-07-01") & date <= as.Date("2005-10-01") ~ "fd_2",
      date >= as.Date("2007-01-01") & date <= as.Date("2010-04-01") ~ "fd_3",
      date >= as.Date("2011-01-01") & date <= as.Date("2015-10-01") ~ "fd_4",
      TRUE ~ "Other"
    ),
    dominance_type = case_when(
      str_detect(fd_period, "^md") ~ "monetary",
      str_detect(fd_period, "^fd") ~ "fiscal",
      TRUE ~ NA
    )
  ) %>%
  mutate(mean_period = mean(fiscal_v_monetary, na.rm = T), .by = "fd_period") %>%
  mutate(mean_dominance = mean(fiscal_v_monetary, na.rm = T), .by = "dominance_type")

indicator_comp <- speech_aggregation %>%
  filter(country == "USA") %>%
  select(year, date, monetary_dominance, fiscal_dominance) %>%
  filter(year <= 2017) %>%
  mutate(
    fiscal_v_monetary = fiscal_dominance / (fiscal_dominance + monetary_dominance)
  ) %>%
  mutate(
    avg = slide_index_dbl(fiscal_v_monetary, date, ~mean(.x, na.rm = T), .before = lubridate::days(182), .after = lubridate::days(182))) %>%
  group_by(date) %>%
  slice(1) %>% # Speeches on the same date have the same values
  ungroup() %>%
  select(date, avg) %>%
  ggplot(aes(y = avg, x = date)) +
  geom_line(linewidth = 0.3) +
  annotate("rect", fill = colors[2], alpha = 0.1,
           xmin = as.Date("2000-10-01"), xmax = as.Date("2003-10-01"),
           ymin = -Inf, ymax = Inf) +
  annotate("rect", fill = colors[2], alpha = 0.1,
           xmin = as.Date("2004-07-01"), xmax = as.Date("2005-10-01"), #
           ymin = -Inf, ymax = Inf) +
  annotate("rect", fill = colors[2], alpha = 0.1,
           xmin = as.Date("2007-01-01"), xmax = as.Date("2010-04-01"), #
           ymin = -Inf, ymax = Inf) +
  annotate("rect", fill = colors[2], alpha = 0.1,
           xmin = as.Date("2011-01-01"), xmax = as.Date("2015-10-01"), #
           ymin = -Inf, ymax = Inf) +
  geom_line(
    data = f_v_period_mean %>% mutate(mean = case_when(dominance_type != "monetary" ~ NA, TRUE ~ mean_dominance)), aes(x = date, y = mean),
    color = colors[1]
  ) +
  geom_line(
    data = f_v_period_mean %>% mutate(mean = case_when(dominance_type != "fiscal" ~ NA, TRUE ~ mean_dominance)), aes(x = date, y = mean),
    color = colors[2]
  ) +
  xlim(min(f_v_period_mean$date), as.Date("2018-01-01")) +
  scale_y_continuous(expand = c(0, 0), n.breaks = 5, limits = c(0, NA), labels = scales::percent) +
  labs(
    x = "Year",
    y = "Share fiscal dominance"
  ) +
  paper_theme()

indicator_comp %>% ggsave(
  filename = "./output/figures/descriptive/indicator_comp.pdf",
  width = default_chart_width,
  height = default_chart_height * 0.8,
  units = "cm",
  device = cairo_pdf
)