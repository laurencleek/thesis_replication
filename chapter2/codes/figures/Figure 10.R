library(tidyverse)
library(readxl)
library(arrow)
library(patchwork)
library(ggridges)

# Load chart settings
source("./codes/functions/chart_themes.R")

# Load variable names, DiD settings and DiD functions
source("./codes/constants/varnames.R")
source("./codes/constants/diff_in_diff_settings.R")
source("./codes/functions/diff_in_diff_functions.R")

# -------------------------------------------------------------------
# LOAD DATA AND TREATMENTS
# -------------------------------------------------------------------

# Load data speech level data
speech_agg <- read_parquet("./data/processed/aggregations/speech_agg.parquet") %>% filter(
  year >= 1997,
  year <= 2023
)

# Load data aggregated by country, year
country_year_agg <- read_parquet("./data/processed/aggregations/country_year_agg.parquet") %>% filter(
  year >= 1997,
  year <= 2023
)

# These are the actual countries we have in the speeches dataset
speeches_countries <- setdiff(speech_agg %>% pull(country) %>% unique(), "")

# Generate binary treatments for the main specification
binary_case_treatments <- generate_treatment(which = "max",
                                             minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                             range_years = TREATMENT_YEARS_RANGE,
                                             balance_panel_strategy = "zero",
                                             dataset = "ro",
                                             allow_multiple_treatments = F,
                                             forward_fill_until = max(TREATMENT_YEARS_RANGE),
                                             limit_countries = speeches_countries)

# Join treatments with speech aggregation
binary_case <- left_join(binary_case_treatments, speech_agg, by = c("country", "year"))

# Some models are on the country year level
country_year_binary_case <- binary_case_treatments %>% left_join(
  country_year_agg, by = c("country", "year"))

# Continueous case:
continueous_case <- generate_treatment(balance_panel_strategy = "zero",
                                       range_years = TREATMENT_YEARS_RANGE,
                                       dataset = "ro",
                                       negative_strategy = "allow",
                                       allow_multiple_treatments = T) %>%
  left_join(speech_agg, by = c("country", "year"))

# This is needed for the distributed lag models.
continueous_case_absolute_intensities <- generate_treatment(balance_panel_strategy = "zero",
                                                            range_years = TREATMENT_YEARS_RANGE,
                                                            cumulate_intensity = "absolute",
                                                            negative_strategy = "allow",
                                                            allow_multiple_treatments = T) %>%
  left_join(speech_agg, by = c("country", "year"))

patchwork::wrap_plots(
  plot_estimates(twfe_freyaldenhoven("inflation",
                                     country_year_binary_case %>% mutate(
                                       inflation = pmin(inflation,
                                                        country_year_binary_case %>%
                                                          pull(inflation) %>%
                                                          quantile(0.99, na.rm = T)) # Windsorize at 99%
                                     ),
                                     leads = EVENT_STUDY_NUMBER_LEADS,
                                     lags = EVENT_STUDY_NUMBER_LAGS),
                 add_tests = T,
                 tests_font_size = 2.15,
                 tests_text_line_width = 0.9,
                 tests_x = 3.5,
                 tests_y = 0.084,
                 add_supt_band = T,
                 ylim = c(-0.11, 0.08),
                 manual_breaks = c(-5, -3, -1, 0, 2, 4, 6, 8, 10, 12)) +
    ggtitle("A. HICP inflation") +
    theme(
      axis.title.y = element_text(margin = margin(t = 0, r = -1, b = 0, l = 0)),
    ),

  plot_estimates(twfe_freyaldenhoven("financial_stress",
                                     country_year_binary_case,
                                     leads = EVENT_STUDY_NUMBER_LEADS,
                                     lags = EVENT_STUDY_NUMBER_LAGS),
                 ylim = c(-0.12, 0.25),
                 add_tests = T,
                 tests_font_size = 2.15,
                 tests_text_line_width = 0.9,
                 tests_x = 3.5,
                 tests_y = 0.257,
                 add_supt_band = T,
                 manual_breaks = c(-5, -3, -1, 0, 2, 4, 6, 8, 10, 12)) +
    ggtitle("B. Financial stress") +
    theme(
      axis.title.y = element_text(margin = margin(t = 0, r = -1, b = 0, l = 7)), # Fix spacing between charts
    )
)

ggsave(
  filename = "output/figures/mechanisms/event_study_hicp_financial.pdf",
  width = panel_chart_width,
  height = default_chart_height * 0.75,
  device = cairo_pdf,
  units = "cm"
)