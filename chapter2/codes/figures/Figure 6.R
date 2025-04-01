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

indicator_over_time <- speech_aggregation %>%
  mutate(
    across(all_of(dom), function(dominance) {
      slide_index_dbl(dominance, date, ~mean(.x, na.rm = T), .before = lubridate::days(182), .after = lubridate::days(182))
    })) %>%
  group_by(date) %>%
  slice(1) %>% # Speeches on the same haave the same values
  ungroup() %>%
  select(date, all_of(dom)) %>%
  pivot_longer(-date, names_to = "dominance", values_to = "value") %>%
  mutate(
    dominance = factor(dominance, levels = dom)
  ) %>%
  ggplot(aes(x = date, y = value, color = dominance)) +
  geom_line(linewidth = 0.35) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE, keyheight = 0.5)) +
  scale_x_date(date_breaks = "5 years", date_labels = "%Y", limits = (c(as.Date("1997-01-01"), as.Date("2023-01-01")))) +
  scale_y_continuous(expand = c(0, 0), n.breaks = 8, limits = c(0, NA), labels = scales::percent) +
  scale_color_discrete(labels = dom_label) +
  labs(color = "", x = "", y = "Share") +
  paper_theme()

ggsave(
  indicator_over_time,
  filename = "output/figures/descriptive/indicator_over_time.pdf",
  width = default_chart_width,
  height = default_chart_height * 0.9,
  device = cairo_pdf,
  units = "cm"
)