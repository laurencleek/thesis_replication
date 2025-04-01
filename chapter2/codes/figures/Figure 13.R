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

# Alternative explanations tests
ecb_added_as_control <- dom[named_dom %in% dominance_to_plot] %>% map(
  function(dom) {
    ecb_as_control <- generate_treatment(which = "max",
                                         minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                         range_years = TREATMENT_YEARS_RANGE,
                                         balance_panel_strategy = "zero",
                                         dataset = "ro",
                                         allow_multiple_treatments = F,
                                         ro_add_ecb_to_control = T,
                                         forward_fill_until = max(TREATMENT_YEARS_RANGE),
                                         limit_countries = speeches_countries) %>%
      left_join(speech_agg, by = c("country", "year"))

    twfe_binary(dom, ecb_as_control, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL, overwrite_model_name = "ECB added as control")
  }
)

first_treatment <- dom[named_dom %in% dominance_to_plot] %>% map(
  function(dom) {
    only_first_treatment_then_drop <- generate_treatment(which = "first",
                                                         minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                                         range_years = TREATMENT_YEARS_RANGE,
                                                         balance_panel_strategy = "zero",
                                                         dataset = "ro",
                                                         allow_multiple_treatments = F,
                                                         forward_fill_until = max(TREATMENT_YEARS_RANGE),
                                                         limit_countries = speeches_countries) %>%
      left_join(speech_agg, by = c("country", "year")) %>%
      filter(
        number_of_increases < 2
      )

    twfe_binary(dom, only_first_treatment_then_drop, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL, overwrite_model_name = "First independence change")
  }
)

without_ea <- dom[named_dom %in% dominance_to_plot] %>% map(
  function(dom) {
    drop_ea <- binary_case %>% filter(currency_code != "EUR")
    twfe_binary(dom, drop_ea, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL, overwrite_model_name = "Drop euro area")
  }
)

plot_dominance_panel(c(without_ea, ecb_added_as_control, first_treatment),
                     custom_limits_y = list(
                       "Monetary dominance" = c(-0.33, 0.17),
                       "Financial dominance" = c(-0.12, 0.22),
                       "Fiscal dominance" = c(-0.06, 0.11)),
                     custom_breaks_y = list(
                       "Monetary dominance" = c(-0.3, -0.15, 0, 0.15),
                       "Financial dominance" = c(-0.1, 0, 0.1, 0.2),
                       "Fiscal dominance" = c(-0.05, 0, 0.05, 0.1)),
                     custom_breaks_x = c(-5, 0, 5, 10),
                     remove_facet_titles = T) %>%
  ggsave(
    filename = "output/figures/did/sample_variations_main.pdf",
    width = panel_chart_width,
    height = panel_chart_height * 0.7,
    device = cairo_pdf,
    units = "cm"
  )