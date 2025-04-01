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

## Compare datasets:
# Generate binary treatments for the main specification
binary_case_treatments_alt_dataset <- generate_treatment(which = "max",
                                             minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                             range_years = TREATMENT_YEARS_RANGE,
                                             balance_panel_strategy = "zero",
                                             dataset = "ga",
                                             ga_indicator = "lvau_garriga",
                                             allow_multiple_treatments = F,
                                             forward_fill_until = max(TREATMENT_YEARS_RANGE),
                                             limit_countries = speeches_countries)

# Join treatments with speech aggregation
binary_case_alt_dataset <- left_join(binary_case_treatments_alt_dataset, speech_agg, by = c("country", "year"))

count <- 1
(named_dom[named_dom %in% dominance_to_plot] %>%
  imap(function(dom, name) {
    limits <- list(
      "monetary_dominance" = c(-0.44, 0.22),
      "financial_dominance" = c(-0.09, 0.16),
      "fiscal_dominance" = c(-0.25, 0.25)
    )[[dom]]

    breaks <- list(
      "monetary_dominance" =c(-0.4, -0.2, 0, 0.2),
      "financial_dominance" = c(-0.075, 0, 0.075, 0.15),
      "fiscal_dominance" = c(-0.2, 0, 0.2)
    )[[dom]]

    plot <- plot_multiple_estimates(list(
      twfe_binary(dom, binary_case, leads = EVENT_STUDY_NUMBER_LEADS,
                  lags = EVENT_STUDY_NUMBER_LAGS, fill = 0, overwrite_model_name = "Romelli (2024)"),
      twfe_binary(dom, binary_case_alt_dataset, leads = EVENT_STUDY_NUMBER_LEADS,
                  lags = EVENT_STUDY_NUMBER_LAGS, fill = 0, overwrite_model_name = "Garriga (2025)")),
                                    dodge_amount = 0.55,
                                    ylim = limits,
                                    ybreaks = breaks,
                                    percent_scale = TRUE
    )
    plot <- plot + ggtitle(str_glue("{toupper(letters[count])}. {name}"))
    count <<- count + 1
    plot
  }) %>%
  patchwork::wrap_plots(ncol = if (export_to_slides) 2 else 1,
                        design = if (export_to_slides) patchwork2x1_design else NULL,
                        guides = "collect") & theme(legend.position = 'bottom'))

ggsave(
    filename = "output/figures/did/dataset_comparison.pdf",
    width = if (export_to_slides) panel_slide_width else default_chart_width + 1,
    height = if (export_to_slides) panel_slide_height else panel_chart_height,
    device = cairo_pdf,
    units = "cm"
)