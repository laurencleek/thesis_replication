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

# Fix strange bug where France and Norway are missing
world <- ne_countries(scale = "small", returnclass = "sf") %>% mutate(
  iso_a3 = case_when(
    iso_a3 == -99 ~ iso_a3_eh,
    TRUE ~ iso_a3
  )
)

# Fix Crimea to be part of Ukraine!
crimea <- world[world$iso_a3 == "RUS", "geometry"][[1]][[1]][[14]]
# Remove Crimea from Russia.
world[world$iso_a3 == "RUS", "geometry"][[1]][[1]][[14]] <- NULL
# Add to Ukraine
world[world$iso_a3 == "UKR", "geometry"][[1]][[1]] <- c(world[world$iso_a3 == "UKR", "geometry"][[1]][[1]], st_polygon(crimea))

event_map <- world %>%
  filter(continent != "Antarctica") %>%
  st_transform(crs = "+proj=robin") %>%
  left_join(events_by_country, join_by("iso_a3")) %>%
  mutate(
    number_of_events = pmin(number_of_events, 6),
    number_of_events = factor(
      number_of_events, levels = 0:6, labels = as.character(seq(0:6), "≥6")
    )
  ) %>%
  ggplot() +
  geom_sf(aes(fill = number_of_events)) +
  #coord_sf(datum = NA) + Remove lines
  scale_fill_brewer(name = "",
                    palette = "Blues",
                    direction = 1,
                    na.value = "gray90",
                    labels = c(0, 1, 2, 3, 4, 5, expression("" >= 6), "no data"),
                    guide = guide_legend(
                      direction = "horizontal",
                      keyheight = unit(2.5, units = "mm"),
                      keywidth = unit(0.75, units = "cm"),
                      title.position = 'top',
                      title.hjust = 0.5,
                      label.hjust = 0.5,
                      nrow = 1,
                      byrow = FALSE,
                      reverse = FALSE,
                      override.aes = list(color = NA),
                      label.position = "bottom")
  ) +
  paper_theme_map()

event_map %>% ggsave(
  filename = "./output/figures/descriptive/event_map.pdf",
  width = panel_chart_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)