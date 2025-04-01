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

# First calculate mean by independence event type
mean_by_change_type <- cbie_changes %>%
  filter(year >= lower_bound, year <= upper_bound) %>%
  mutate(
    change_type = case_when(
      independence_increase ~ "increase",
      independence_decrease ~ "decrease",
      TRUE ~ NA
    )
  ) %>%
  filter(!is.na(change_type)) %>%
  summarise(
    mean_magnitude = mean(intensity),
    .by = "change_type"
  )

all_independece_changes_data <- intersection_of_cbi_and_speeches %>%
  mutate(
    independence_increase = (cbie_index - dplyr::lag(cbie_index)) > 0,
    independence_decrease = (cbie_index - dplyr::lag(cbie_index)) < 0,
    .by = "country"
  ) %>%
  filter(year >= lower_bound, year <= upper_bound) %>%
  summarise(
    increases = sum(independence_increase, na.rm = T),
    decreases = -sum(independence_decrease, na.rm = T),
    .by = "year"
  ) %>%
  arrange(year) %>%
  pivot_longer(
    -year, values_to = "ind_change", names_to = "type"
  ) %>%
  mutate(
    type = factor(type, levels = c("increases", "decreases"), labels = c("CBI increase", "CBI decrease"))
  )

relevant_independence_changes_data <-
  generate_treatment("ro",
                     lower_bound:upper_bound,
                     limit_countries = included_countries_speeches_and_romelli,
                     minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                     negative_strategy = "zero",
                     allow_multiple_treatments = FALSE,
                     balance_panel_strategy = "zero") %>%
    summarise(
      increases = sum(intensity > 0, na.rm = T),
      decreases = sum(intensity < 0, na.rm = T),
      .by = "year"
    ) %>%
    arrange(year) %>%
    pivot_longer(
      -year, values_to = "ind_change", names_to = "type"
    ) %>%
    mutate(
      type = factor(type, levels = c("increases", "decreases"), labels = c("CBI increase", "CBI decrease"))
    )

# Create histogram of independence changes
distribution_of_independence_changes <- cbie_changes %>%
  filter(year >= lower_bound, year <= upper_bound, change == 1) %>%
  left_join(
    generate_treatment("ro",
                       lower_bound:upper_bound,
                       limit_countries = included_countries_speeches_and_romelli,
                       minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                       negative_strategy = "zero",
                       allow_multiple_treatments = FALSE,
                       balance_panel_strategy = "zero") %>%
      filter(treating == 1) %>%
      select(country, year) %>%
      mutate(
        relevance = "relevant_ind_change"
      ), by = join_by(iso_a3 == country, year == year)
  ) %>%
  mutate(
    relevance = replace_na(relevance, "irrelevant_ind_change"),
    change_type = case_when(
      independence_increase ~ "increase",
      independence_decrease ~ "decrease",
      TRUE ~ NA
    ),
    change_type = factor(change_type, levels = c("increase", "decrease"), labels = c("CBI increase", "CBI decrease"))
  ) %>%
  select(intensity, relevance, change_type) %>%
  ggplot(aes(x = intensity, color = change_type, fill = change_type, pattern = relevance, alpha = relevance)) +
  geom_histogram_pattern(pattern_color = NA,
                         pattern_fill = "white",
                         breaks = seq(-0.2, 0.6, 0.025),
                         linewidth = 0.3,
                         pattern_spacing = 0.01, pattern_density = 0.07) +
  scale_alpha_manual(values = c(HIST_FILL_ALPHA * 0.9, HIST_FILL_ALPHA * 1.1), guide = "none") +
  scale_pattern_manual(values = c('stripe', "none"), guide = "none") +
  geom_vline(xintercept = 0, linewidth = 0.35) +
  geom_vline(
    data = mean_by_change_type %>% mutate(change_type = factor(change_type, levels = c("increase", "decrease"), labels = c("CBI increase", "CBI decrease"))),
    aes(xintercept = mean_magnitude, color = change_type),
    linetype = "dashed", show.legend = FALSE) +
  paper_theme() +
  labs(x = "Change magnitude", y = "Count", color = "", fill = "", alpha = "", pattern = "") +
  scale_y_continuous(expand = c(0, 0), n.breaks = 5, limits = c(0, 30)) +
  guides(fill = guide_legend(override.aes = list(pattern = "none", alpha = HIST_FILL_ALPHA)))

# Create a single graph of the relevant independence events.
independence_events_rel_and_irrel <- all_independece_changes_data %>%
  rename(
    any_ind_change = "ind_change"
  ) %>%
  left_join(
    relevant_independence_changes_data %>% rename(
      relevant_ind_change = "ind_change"
    ),
    by = c("year", "type")
  ) %>%
  mutate(
    irrelevant_ind_change = any_ind_change - relevant_ind_change
  ) %>%
  select(year, type, relevant_ind_change, irrelevant_ind_change) %>%
  pivot_longer(c(relevant_ind_change, irrelevant_ind_change), names_to = "relevance") %>%
  ggplot(aes(x = year, y = value, color = type, fill = type, pattern = relevance, alpha = relevance)) +
  geom_bar_pattern(stat = "identity", pattern_color = NA, pattern_fill = "white", linewidth = 0.35, pattern_spacing = 0.01, pattern_density = 0.07, show.legend = FALSE) +
  geom_hline(yintercept = 0, linewidth = 0.06) +
  scale_pattern_manual(values = c('stripe', "none"), guide = "none") +
  scale_alpha_manual(values = c(HIST_FILL_ALPHA * 0.9, HIST_FILL_ALPHA * 1.1), guide = "none") +
  scale_x_continuous() +
  scale_y_continuous(expand = c(0, 0), breaks = c(-5, 0, 10, 20), limits = c(-5, 20), labels = c(5, 0, 10, 20)) +
  labs(fill = "", color = "", alpha = "", pattern = "", x = "Year", y = "Number of events") +
  paper_theme()

wrap_plots(
  distribution_of_independence_changes + ggtitle("A. Histogram of change magnitudes"),
  independence_events_rel_and_irrel + ggtitle("B. CBI events over time"),
  guides = "collect"
) &
  theme(
    legend.position = "bottom",
    legend.key.size = unit(0.5, "lines"),
    legend.margin = margin(-5, 0, -5, 0, "pt"),
  )

ggsave(
  filename = "output/figures/descriptive/distribution_of_independence_changes.pdf",
  width = panel_chart_width,
  height = default_chart_height * 0.8,
  device = cairo_pdf,
  units = "cm"
)