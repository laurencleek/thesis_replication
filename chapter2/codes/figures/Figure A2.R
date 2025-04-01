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

# Try different specifications accounting for treatment intensity:
count <- 1

(named_dom[named_dom %in% dominance_to_plot] %>%
  imap(function(dom, name) {

    limits <- list(
      "monetary_dominance" = c(-1.2, 0.6),
      "financial_dominance" = c(-0.57, 0.57),
      "fiscal_dominance" = c(-0.25, 0.25)
    )[[dom]]

    breaks <- list(
      "monetary_dominance" = c(-1, -0.5, 0, 0.5),
      "financial_dominance" = c(-0.5, -0.25, 0, 0.25, 0.5),
      "fiscal_dominance" = c(-0.2, 0, 0.2)
    )[[dom]]

    plot <- plot_multiple_estimates(list(
      twfe_binary(dom, binary_case, continueous_treatments = TRUE, leads = EVENT_STUDY_NUMBER_LEADS,
                  lags = EVENT_STUDY_NUMBER_LAGS, fill = 0, overwrite_model_name = "Accounting for intensity"),
      twfe_binary(dom, continueous_case, continueous_treatments = TRUE, leads = EVENT_STUDY_NUMBER_LEADS,
                  lags = EVENT_STUDY_NUMBER_LAGS, fill = 0, overwrite_model_name = "Full variation")),
                                    dodge_amount = 0.55,
                                    ylim = limits,
                                    ybreaks = breaks,
                                    percent_scale = FALSE
    )
    plot <- plot + ggtitle(str_glue("{toupper(letters[count])}. {name}"))
    count <<- count + 1
    plot
  }) %>%
  patchwork::wrap_plots(ncol = if (export_to_slides) 2 else 1,
                        design = if (export_to_slides) patchwork2x1_design else NULL,
                        guides = "collect") & theme(legend.position = 'bottom')) %>%
  ggsave(
    filename = "output/figures/did/dominance_combined_continuous.pdf",
    width = if (export_to_slides) panel_slide_width else default_chart_width + 1,
    height = if (export_to_slides) panel_slide_height else panel_chart_height,
    device = cairo_pdf,
    units = "cm"
  )