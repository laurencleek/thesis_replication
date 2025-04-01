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

# Distribution of speeches:
distribution_of_speeches <- speech_aggregation %>%
  mutate(
    country_group = case_when(
      country == "USA" ~ "Federal reserve system",
      currency_code == "EUR" ~ "Euro area",
      advanced == 1 ~ "Other advanced",
      advanced == 0 ~ "Other emerging and developing",
      TRUE ~ "Other institutions"
    )
  ) %>%
  summarise(
    count = n(),
    .by = c("country_group", "year")
  ) %>%
  mutate(
    country_group = factor(country_group, levels = c("Euro area", "Federal reserve system", "Other advanced", "Other emerging and developing", "Other institutions"))
  ) %>%
  ggplot(aes(x = year, y = count, fill = country_group, color = country_group)) +
  geom_bar(stat = "identity", width = 0.75) +
  scale_y_continuous(expand = c(0, 0), n.breaks = 5, limits = c(0, 1200)) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE, keyheight = 0)) +
  labs(fill = "", color = "", x = "Year", y = "Number of speeches") +
  paper_theme()

ggsave(
  distribution_of_speeches,
  filename = "output/figures/descriptive/distribution_of_speeches.pdf",
  width = default_chart_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)