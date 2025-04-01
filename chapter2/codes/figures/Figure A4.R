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

cbis <- c("cbie_index", "cbie_board", "cbie_policy", "cbie_obj", "cbie_lending", "cbie_finindep", "cbie_report")
cbi_names <- c("Overall", "Board", "Monetary Policy", "Objectives", "Lending to Gov.", "Financial Indep.", "Reporting")

# Effects by CBI indicator
coef_plot_twfe <- expand.grid(dominance = dom[named_dom %in% dominance_to_plot], cbi_indicator = cbis) %>%
  pmap_dfr(
    function(dominance, cbi_indicator) {
      treatment_indicators <- generate_treatment(which = "max",
                                                 minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                                 range_years = TREATMENT_YEARS_RANGE,
                                                 balance_panel_strategy = "zero",
                                                 dataset = "ro",
                                                 ro_indicator = as.character(cbi_indicator),
                                                 allow_multiple_treatments = F,
                                                 limit_countries = speeches_countries)

      combined_dataset <- left_join(treatment_indicators, speech_agg, by = c("country", "year"))

      twfe_binary(dominance, combined_dataset, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
        .$estimated_model %>%
        aggregate_lags(weight_by_occurnace = T, estimation_data = combined_dataset) %>%
        mutate(
          dominance = dominance,
          cbi_indicator = cbi_indicator
        )
    }
  ) %>%
  mutate(
    conf.high = coef + 1.96 * se,
    conf.low = coef - 1.96 * se,
    cbi_indicator = ordered(cbi_indicator, cbis, labels = cbi_names)
  ) %>%
  ggplot(
    aes(y = cbi_indicator, x = coef, color = dominance, shape = dominance, fill = dominance)
  ) +
  geom_vline(xintercept = 0, linewidth = 0.25) +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high), size = 0.2, position = position_dodge(width = 0.4)) +
  paper_theme() +
  theme(
    panel.grid.major.x = element_line(colour = "grey75", linetype = "dotted", linewidth = 0.25),
    panel.grid.major.y = element_blank(),
  ) +
  labs(
    color = "",
    x = "CBI coeffcient",
    shape = "",
    y = "",
    fill = ""
  ) +
  scale_color_discrete(labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance")) +
  scale_fill_discrete(labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance")) +
  scale_y_discrete(limits = rev) +
  scale_x_continuous(limits = c(-0.4, 0.17), breaks = c(-0.3, -0.15, 0, 0.15)) +
  discrete_scale('shape', 'shape_d', function(n) c(21, 23, 22)[seq_len(n)], labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance"))

coef_plot_gardner <- expand.grid(dominance = dom[named_dom %in% dominance_to_plot], cbi_indicator = cbis) %>%
  pmap_dfr(
    function(dominance, cbi_indicator) {
      treatment_indicators <- generate_treatment(which = "max",
                                                 minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                                 range_years = TREATMENT_YEARS_RANGE,
                                                 balance_panel_strategy = "zero",
                                                 dataset = "ro",
                                                 ro_indicator = as.character(cbi_indicator),
                                                 allow_multiple_treatments = F,
                                                 limit_countries = speeches_countries)

      combined_dataset <- left_join(treatment_indicators, speech_agg, by = c("country", "year"))

      # Estiamte using gardner
      broom::tidy(did2s(
        combined_dataset,
        yname = dominance, first_stage = ~0 | country + year,
        second_stage = ~treated,
        treatment = "treated",
        cluster_var = "country"
      )) %>% mutate(
        dominance = dominance,
        cbi_indicator = cbi_indicator
      )
    }
  ) %>%
  mutate(
    coef = estimate,
    conf.low = estimate - 1.96 * std.error,
    conf.high = estimate + 1.96 * std.error,
    cbi_indicator = ordered(cbi_indicator, cbis, labels = cbi_names)
  ) %>%
  ggplot(
    aes(y = cbi_indicator, x = coef, color = dominance, shape = dominance, fill = dominance)
  ) +
  geom_vline(xintercept = 0, linewidth = 0.25) +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high), size = 0.2, position = position_dodge(width = 0.4)) +
  paper_theme() +
  theme(
    panel.grid.major.x = element_line(colour = "grey75", linetype = "dotted", linewidth = 0.25),
    panel.grid.major.y = element_blank(),
  ) +
  labs(
    color = "",
    shape = "",
    x = "CBI coeffcient",
    y = "",
    fill = ""
  ) +
  scale_shape(labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance")) +
  scale_color_discrete(labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance")) +
  scale_fill_discrete(labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance")) +
  scale_y_discrete(limits = rev) +
  scale_x_continuous(limits = c(-0.4, 0.17), breaks = c(-0.3, -0.15, 0, 0.15)) +
  discrete_scale('shape', 'shape_d', function(n) c(21, 23, 22)[seq_len(n)], labels = c("Monetary dominance", "Financial dominance", "Fiscal dominance"))

(patchwork::wrap_plots(coef_plot_twfe + ggtitle("A. Two-way fixed effects"), coef_plot_gardner + ggtitle("B. Gardner (2022)"), ncol = 2, guides = "collect") &
  theme(legend.position = 'bottom')) %>%
  ggsave(
    filename = "output/figures/did/coef_plot_did_combined.pdf",
    width = panel_chart_width,
    height = default_chart_height,
    device = cairo_pdf,
    units = "cm"
  )