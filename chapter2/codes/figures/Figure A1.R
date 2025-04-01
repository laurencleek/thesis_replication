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

# Robustness: Linear trends and macroeconomic controls
monetary_dominance_twfe_rob <- plot_multiple_estimates(list(
  twfe_binary("monetary_dominance", binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL, controls = c("gdp_real_growth", "inflation", "structural_balance_pgdp"), overwrite_model_name = "Control for macroeconomic indicators"),
  twfe_binary("monetary_dominance", binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, unit_trends = T, overwrite_model_name = "Country specific linear trends")),
                                                       ylim = c(-0.4, 0.24),
                                                       ybreaks = c(-0.4, -0.2, 0, 0.2),
                                                       dodge_amount = 0.55,
                                                       zero_line_value = binary_case %>%
                                                         filter(time_post_treat == -1) %>%
                                                         pull(monetary_dominance) %>%
                                                         mean(na.rm = T)
)

financial_dominance_twfe_rob <- plot_multiple_estimates(list(
  twfe_binary("financial_dominance", binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL, controls = c("gdp_real_growth", "inflation", "structural_balance_pgdp"), overwrite_model_name = "Control for macroeconomic indicators"),
  twfe_binary("financial_dominance", binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, unit_trends = T, overwrite_model_name = "Country specific linear trends")),
                                                        ylim = c(-0.1, 0.22),
                                                        ybreaks = c(-0.1, 0, 0.1, 0.2),
                                                        dodge_amount = 0.55,
                                                        zero_line_value = binary_case %>%
                                                          filter(time_post_treat == -1) %>%
                                                          pull(financial_dominance) %>%
                                                          mean(na.rm = T)
)

fiscal_dominance_twfe_rob <- plot_multiple_estimates(list(
  twfe_binary("fiscal_dominance", binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL, controls = c("gdp_real_growth", "inflation", "structural_balance_pgdp"), overwrite_model_name = "Control for macroeconomic indicators"),
  twfe_binary("fiscal_dominance", binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, unit_trends = T, overwrite_model_name = "Country specific linear trends")),
                                                     ylim = c(-0.07, 0.11),
                                                     ybreaks = c(-0.05, 0, 0.05, 0.1),
                                                     dodge_amount = 0.55,
                                                     zero_line_value = binary_case %>%
                                                       filter(time_post_treat == -1) %>%
                                                       pull(fiscal_dominance) %>%
                                                       mean(na.rm = T)
)

(patchwork::wrap_plots(
  list(
  monetary_dominance = monetary_dominance_twfe_rob + ggtitle("A. Monetary dominance"),
  financial_dominance = financial_dominance_twfe_rob + ggtitle("B. Financial dominance"),
  fiscal_dominance = fiscal_dominance_twfe_rob + ggtitle("C. Fiscal dominance")
  )[dominance_to_plot],
  ncol = if (export_to_slides) 2 else 1,
  design = if (export_to_slides) patchwork2x1_design else NULL, guides = "collect") &
  theme(legend.position = 'bottom')) %>%
  ggsave(
    filename = "./output/figures/did/dominance_combined_twfe_robustness.pdf",
    width = if (export_to_slides) panel_slide_width else default_chart_width + 1,
    height = if (export_to_slides) panel_slide_height else panel_chart_height,
    device = cairo_pdf,
    units = "cm"
  )