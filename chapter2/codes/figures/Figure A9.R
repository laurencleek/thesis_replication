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

#---------------------------------
# Comparison of Romelli and Garriga:
#---------------------------------

# Compare country coverage in the datasets:
## Look at the raw data:
ro_indicator <- "cbie_index"
ga_indicator <- "lvau_garriga"

data_ga <- read_excel("./data/input/metadata/central_bank/CBI_2025_websiteGarriga.xlsx", sheet = "Sheet1")  %>% mutate(
        ccodewb = case_match(ccodewb, 891 ~ 688, 890 ~ 688, .default = ccodewb),
        country = countrycode(ccodewb, "iso3n", "iso3c")
) %>%
  mutate(ga_indicate = get(ga_indicator)) %>%
  select(ga_indicate, year , country)

data_ro <- read_excel("./data/input/metadata/central_bank/CBIData_Romelli_2024.xlsx", sheet = "CBI data") %>%
  select(-country) %>%
  rename(country = iso_a3) %>%
  mutate(ro_indicate = get(ro_indicator)) %>%
  select(ro_indicate, year , country)

combined_ro_ga <- expand.grid(country = speeches_countries,year = (lower_bound-1):upper_bound) %>%
    left_join(data_ga,  by = join_by(country == country, year == year)) %>%
    left_join(data_ro,  by = join_by(country == country, year == year)) %>%
  tibble() %>% arrange(
  country, year
) %>% mutate(
  ga_increase = ga_indicate > lag(ga_indicate),
  ga_decrease = (ga_indicate < lag(ga_indicate)) * -1,
  ro_increase = ro_indicate > lag(ro_indicate),
  ro_decrease = (ro_indicate < lag(ro_indicate)) * -1,
  .by = "country"
) %>%
  left_join(
    generate_treatment("ro",
                     lower_bound:upper_bound,
                     minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                     negative_strategy = "zero",
                     allow_multiple_treatments = FALSE,
                     balance_panel_strategy = "zero") %>% select(
      country, year, treating
    ) %>% mutate(
      ro_relevant_increase = treating,
      ro_relevant_decrease = (treating == -1) * -1
    ), by = join_by(country == country ,year == year)
  )  %>%
  select(-treating) %>%
  left_join(
    generate_treatment("ga",
                     lower_bound:upper_bound,
                     minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                     negative_strategy = "zero",
                     allow_multiple_treatments = FALSE,
                     balance_panel_strategy = "zero") %>% select(
      country, year, treating
    ) %>% mutate(
      ga_relevant_increase = treating,
      ga_relevant_decrease = (treating == -1) * -1
    ), by = join_by(country == country ,year == year)
  ) %>%
  select(-treating)

coverage_ro_ga <-
  combined_ro_ga %>%
    summarise(
      covered_ga = any(!is.na(ga_indicate)),
      covered_ro = any(!is.na(ro_indicate)),
      .by = country
    ) %>%
  mutate(
    coverage = case_when(
      covered_ga & covered_ro ~ "covered_both",
      covered_ro & !covered_ga ~ "covered_ro_only",
      covered_ga & !covered_ro ~ "covered_ga_only",
      !covered_ga & !covered_ro ~ "covered_none"
    )
  )

combined_ro_ga <- combined_ro_ga %>% left_join(
  coverage_ro_ga %>% select(country, coverage), by = "country"
)

relevant_and_irrelevant_event_frequency <- combined_ro_ga %>%
  filter(coverage == "covered_both") %>%
  select(-coverage) %>%  pivot_longer(cols = starts_with("ga_") | starts_with("ro_"),
               names_to = c("dataset", "type"),
               names_sep = "(?<=ga|ro)_",
               values_to = "value") %>%
  filter(type != "indicate") %>%
  mutate(
    relevance =
      case_when(
       str_detect(type,"relevant") ~ "relevant_ind_change",
       TRUE ~ "irrelevant_ind_change"
      ),
    type = case_when(
      type %in% c("relevant_increase", "increase") ~ "CBI increase",
      type %in% c("relevant_decrease", "decrease") ~ "CBI decrease"
    )
  ) %>%  summarise(
  value = sum(value, na.rm = T),
  .by = c("year", "type", "relevance", "dataset")
  ) %>%
  filter(year >= lower_bound, year <= upper_bound)

garriga_events <-
  relevant_and_irrelevant_event_frequency %>%
  filter(dataset == "ga") %>%
  mutate(
    type = factor(type, levels = c("CBI increase", "CBI decrease"))
  ) %>%
  ggplot(aes(x = year, y = value, color = type, fill = type, pattern = relevance, alpha = relevance)) +
  geom_bar_pattern(stat = "identity", pattern_color = NA, pattern_fill = "white", linewidth = 0.35, pattern_spacing = 0.01, pattern_density = 0.07, show.legend = FALSE) +
  geom_hline(yintercept = 0, linewidth = 0.06) +
  scale_pattern_manual(values = c('stripe', "none"), guide = "none") +
  scale_alpha_manual(values = c(HIST_FILL_ALPHA * 0.9, HIST_FILL_ALPHA * 1.1), guide = "none") +
  scale_x_continuous() +
  scale_y_continuous(expand = c(0, 0), breaks = c(-5, 0, 10, 20), limits = c(-5, 20), labels = c(5, 0, 10, 20)) +
  labs(fill = "", color = "", alpha = "", pattern = "", x = "Year", y = "Number of events") +
  geom_text(aes(x = 2010,
                label = str_glue("Increases = {number_of_increases} ({number_of_relevant_increases})",
                                 number_of_relevant_increases = relevant_and_irrelevant_event_frequency %>%
                                   filter(dataset == "ga", type == "CBI increase", relevance == "relevant_ind_change") %>%
                                   pull(value) %>%
                                   sum(),
                                 number_of_increases = relevant_and_irrelevant_event_frequency %>%
                                   filter(dataset == "ga", type == "CBI increase") %>%
                                   pull(value) %>%
                                   sum()
                ),
                y = 19.2), color = colors[1], alpha = 1, hjust = 0, size = 2, check_overlap = TRUE) +
  geom_text(aes(x = 2010,
                label = str_glue("Decreases = {number_of_decreases}",
                                 number_of_decreases = relevant_and_irrelevant_event_frequency %>%
                                   filter(dataset == "ga", type == "CBI decrease") %>%
                                   pull(value) %>%
                                   sum() * (-1)
                ),
                y = 18.2), color = colors[2], alpha = 1, hjust = 0, size = 2, check_overlap = TRUE) +
  paper_theme()

romelli_events <-
  relevant_and_irrelevant_event_frequency %>%
  filter(dataset == "ro") %>%
  mutate(
    type = factor(type, levels = c("CBI increase", "CBI decrease"))
  ) %>%
  ggplot(aes(x = year, y = value, color = type, fill = type, pattern = relevance, alpha = relevance)) +
  geom_bar_pattern(stat = "identity", pattern_color = NA, pattern_fill = "white", linewidth = 0.35, pattern_spacing = 0.01, pattern_density = 0.07, show.legend = TRUE) +
  geom_hline(yintercept = 0, linewidth = 0.06) +
  scale_pattern_manual(values = c('stripe', "none"), guide = "none") +
  scale_alpha_manual(values = c(HIST_FILL_ALPHA * 0.9, HIST_FILL_ALPHA * 1.1), guide = "none") +
  scale_x_continuous() +
  scale_y_continuous(expand = c(0, 0), breaks = c(-5, 0, 10, 20), limits = c(-5, 20), labels = c(5, 0, 10, 20)) +
  labs(fill = "", color = "", alpha = "", pattern = "", x = "Year", y = "Number of events")  +
  geom_text(aes(x = 2010,
                label = str_glue("Increases = {number_of_increases} ({number_of_relevant_increases})",
                                 number_of_relevant_increases = relevant_and_irrelevant_event_frequency %>%
                                   filter(dataset == "ro", type == "CBI increase", relevance == "relevant_ind_change") %>%
                                   pull(value) %>%
                                   sum(),
                                 number_of_increases = relevant_and_irrelevant_event_frequency %>%
                                   filter(dataset == "ro", type == "CBI increase") %>%
                                   pull(value) %>%
                                   sum()
                ),
                y = 19.2), color = colors[1], alpha = 1, hjust = 0, size = 2, check_overlap = TRUE) +
  geom_text(aes(x = 2010,
                label = str_glue("Decreases = {number_of_decreases}",
                                 number_of_decreases = relevant_and_irrelevant_event_frequency %>%
                                   filter(dataset == "ro", type == "CBI decrease") %>%
                                   pull(value) %>%
                                   sum() * (-1)
                ),
                y = 18.2), color = colors[2], alpha = 1, hjust = 0, size = 2, check_overlap = TRUE) +
  paper_theme()

wrap_plots(
  garriga_events + ggtitle("A. CBI events (Garriga, 2025)"),
  romelli_events + ggtitle("B. CBI events (Romelli, 2024)"),
  guides = "collect"
) & theme(
    legend.position = "bottom",
    legend.key.size = unit(0.5, "lines"),
    legend.margin = margin(-5, 0, -5, 0, "pt"),
  )

ggsave(
  filename = "output/figures/descriptive/cbi_changes_over_time_datasetcomp.pdf",
  width = panel_chart_width,
  height = default_chart_height * 0.8,
  device = cairo_pdf,
  units = "cm"
)