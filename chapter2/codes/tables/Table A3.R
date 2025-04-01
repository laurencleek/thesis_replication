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

# All cbi subindicators
cbi_sub_ind <- c("cbie_index", "cbie_board", "cbie_policy", "cbie_obj", "cbie_lending", "cbie_finindep", "cbie_report")

# In addition to the speech aggregation directly load the full Romelli dataset
intersection_of_cbi_and_speeches <- read_excel("./data/input/metadata/central_bank/CBIData_Romelli_2024.xlsx", sheet = "CBI data") %>%
  left_join(speech_aggregation %>%
              select(central_bank, country) %>%
              group_by(country) %>%
              slice(1), by = join_by(iso_a3 == country)) %>%
  filter(!is.na(central_bank))

# Countries available in both
included_countries_speeches_and_romelli <- intersection_of_cbi_and_speeches %>% distinct(iso_a3) %>% pull(iso_a3)

# All CBI chagnes
cbie_changes <- intersection_of_cbi_and_speeches %>% mutate(
  independence_increase = (cbie_index - dplyr::lag(cbie_index)) > 0,
  independence_decrease = (cbie_index - dplyr::lag(cbie_index)) < 0,
  intensity = (cbie_index - dplyr::lag(cbie_index)),
  change = intensity != 0,
  across(all_of(cbi_sub_ind), ~(.x - dplyr::lag(.x)), .names = "intensity_{.col}"),
  .by = "country"
)

# Helper function remove na and take the first element if there are more than one
rem_na <- function(r) {
  r <- r[!is.na(r)]
  return(
    r[1]
  )
}

# Calculate events by country
events_by_country <- cbie_changes %>%
  filter(year >= lower_bound, year <= upper_bound) %>%
  summarise(
    number_of_events = sum(change, na.rm = T),
    number_of_increases = sum(independence_increase, na.rm = T),
    number_of_decreases = sum(independence_decrease, na.rm = T),
    largest_event_magnitude = rem_na(intensity[intensity == max(intensity, na.rm = T) & intensity > 0]),
    year_of_largest_event = rem_na(year[intensity == max(intensity, na.rm = T) & intensity > 0]),
    .by = "iso_a3"
  ) %>%
  arrange(-number_of_events)

# CBI coverage in Romelli
options(knitr.kable.NA = '-')
cbie_changes %>%
  summarise(first_year = min(year), .by = "iso_a3") %>%
  left_join(
    events_by_country, by = "iso_a3"
  ) %>%
  left_join(
    speech_aggregation %>% summarise(
      number_of_speeches = n(),
      .by = "country"
    ),
    by = join_by(iso_a3 == country)
  ) %>%
  select(
    iso_a3,
    number_of_speeches,
    first_year,
    number_of_increases,
    number_of_decreases,
    largest_event_magnitude,
    year_of_largest_event
  ) %>%
  arrange(
    -number_of_speeches
  ) %>%
  rename(
    "Country" = "iso_a3",
    "CBI coverage" = "first_year",
    "Speeches" = "number_of_speeches",
    "Increases" = "number_of_increases",
    "Decreases" = "number_of_decreases",
    "Magnitude" = "largest_event_magnitude",
    "Year" = "year_of_largest_event"
  ) %>%
  mutate(
    Country = case_when(
      is.na(Magnitude) | Magnitude < 0.05 ~ str_glue("{Country}*"),
      TRUE ~ Country
    )
  ) %>%
  kbl(booktabs = T,
      longtable = T,
      format = "latex",
      linesep = "",
      align = "lcccccc",
      caption = "Dataset coverage\\label{tab:dataset_overview}",
      digits = 2) %>%
  add_header_above(c(" " = 3, "Number of events since 1985" = 2, "Largest event since 1985" = 2)) %>%
  kable_styling(latex_options = "repeat_header") %>%
  save_kable("./output/tables/descriptive/dataset_overview.tex")