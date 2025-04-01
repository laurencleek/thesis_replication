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

named_dom[named_dom %in% dominance_to_plot] %>% map_dfr(
  function(dom) {
    twfe_binary(dom,
                                       binary_case %>%
                                         mutate(
                                           audience = factor(audience, levels = audiences, labels = audiences_labels)
                                         ) %>%
                                         group_by(country) %>%
                                         ungroup(),
                                       interact = "audience", leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL
    ) %>%
      .$estimated_model %>%
      aggregate_lags(estimation_data = binary_case, weight_by_occurnace = T) %>%
      mutate(
        labels = audiences_labels,
        dominance = dom
      )
  }
) %>%
  mutate(
    coef = case_when(
      abs(t_stat) > 2.58 ~ paste0(format(round(coef, 4), nsmall = 4), "***"),
      abs(t_stat) > 1.96 ~ paste0(format(round(coef, 4), nsmall = 4), "**"),
      abs(t_stat) > 1.64 ~ paste0(format(round(coef, 4), nsmall = 4), "*"),
      TRUE ~ as.character(format(round(coef, 4), nsmall = 4))
    ),
    se = str_glue("({rounded_se})", rounded_se = as.character(format(round(se, 4), nsmall = 4)))
  ) %>%
  select(dominance, labels, se, coef) %>%
  pivot_longer(c(coef, se), names_to = "se_or_coef", values_to = "values") %>%
  pivot_wider(names_from = dominance, values_from = "values") %>%
  select(
    -se_or_coef
  ) %>%
  mutate(
    labels = replace(labels, seq_along(labels) %% 2 == 0, ""),
    .by = "labels"
  ) %>%
  kbl(format = "latex", booktabs = T, align = c("l", rep('c', length(dominance_to_plot))), col.names = c(" ", names(named_dom[named_dom %in% dominance_to_plot]))) %>%
  save_kable("./output/tables/did/effect_by_audience.tex")