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

Mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

Unique <- function(x) {
  x <- x[!is.na(x)]
  unique(x)
}

# Desc table. Could add more variables here but some of the modes are really long and the table doesnt fit anymore :(
#
desc_table <- speech_aggregation %>%
   mutate(country = na_if(country, ""))
 bind_rows(
   desc_table %>% select(central_bank, date, country, speaker, country, location, longitude, latitude) %>%
     summarise(across(everything(), ~ as.character(sum(!is.na(.x))))) %>% mutate(stat = "n"),
   desc_table %>% select(central_bank, date, country, speaker, country, location, longitude, latitude) %>%
     summarise(across(everything(), ~ as.character(round(sum(is.na(.x) / n()) * 100, 2)))) %>% mutate(stat = "share_na"),
   desc_table %>% select(central_bank, date, country, speaker, country, location, longitude, latitude) %>%
     summarise(across(everything(), ~ as.character(length(Unique(.x))))) %>% mutate(stat = "unique"),
   desc_table %>% select(central_bank, date, country, speaker, country, location, longitude, latitude) %>%
     summarise(across(everything(), ~ as.character(Mode(.x)))) %>% mutate(stat = "mode"),
   desc_table %>% select(central_bank, date, country, speaker, country, location, longitude, latitude) %>%
     summarise(across(everything(), ~ as.character(sum(Mode(.x) == .x, na.rm = T)))) %>% mutate(stat = "mode_n")
 ) %>%
    pivot_longer(-stat) %>%
    pivot_wider(names_from=stat, values_from=value) %>%
   rename("Variable" = "name",
          "Missing (%)" = "share_na",
          "N" = "n",
          "Unique values" = "unique",
          "Mode" = "mode",
          "Mode Freq." = "mode_n") %>%
   kbl(booktabs = T, format = "latex") %>%
   save_kable("./output/tables/descriptive/speech_metadata_descriptives.tex")