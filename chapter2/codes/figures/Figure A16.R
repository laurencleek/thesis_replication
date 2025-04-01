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

# 66 countres are included in both
combined_financial_dominance_and_pressures <- read_excel("./data/input/metadata/country/FSI_Dataset.xlsx", sheet = "T1") %>%
  mutate(
    year_num = as.numeric(str_extract(year, "\\d{4}")),
    quarter_month = as.numeric(str_match(year, "q(\\d)$")[, 2]) * 3 - 2,
    date = ymd(paste(year_num, quarter_month, "1", sep = "-"))
  ) %>%
  pivot_longer(-c(year_num, quarter_month, date, year), names_to = "country", values_to = "financial_stress") %>%
  select(date, country, financial_stress) %>%
  right_join(
    speech_aggregation %>%
      select(date, country, financial_dominance) %>%
      mutate(
        date = ceiling_date(date, "quarter")
      ) %>%
      summarise(
        financial_dominance = mean(financial_dominance, na.rm = T),
        .by = c("country", "date")
      ),
    by = join_by(country == country, date == date)) %>%
  summarise(
    financial_stress = mean(financial_stress, na.rm = T),
    financial_dominance = mean(financial_dominance, na.rm = T),
    .by = "date"
  )

dominance_vs_pressure_chart <- combined_financial_dominance_and_pressures %>%
  pivot_longer(c(financial_stress, financial_dominance), names_to = "indicator", values_to = "amount") %>%
  ggplot(aes(x = date, y = amount, color = indicator)) +
  geom_line() +
  scale_y_continuous(expand = c(0, 0), n.breaks = 4, limits = c(0, NA), labels = scales::percent) +
  scale_x_date(limits = as.Date(c('1997-01-01', '2022-01-01'))) +
  paper_theme() +
  labs(color = "", x = "Year (quaterly observations)", y = "Amount") +
  scale_color_discrete(labels = c("Financial dominance", "Financial stress")) +
  annotate("rect", fill = "grey60", alpha = 0.1,
           xmin = as.Date("2018-10-01"), xmax = as.Date('2022-01-01'),
           ymin = -Inf, ymax = Inf)

dominance_vs_pressure_chart %>% ggsave(
  filename = "./output/figures/descriptive/dominance_vs_pressure.pdf",
  width = default_chart_width,
  height = default_chart_height * 0.9,
  units = "cm",
  device = cairo_pdf
)