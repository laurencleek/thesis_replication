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

expand.grid(estimator = c("twfe", "gardner"), supervision_event =c("maro_supervision", "cbie_policy_q3"), increase = c(TRUE, FALSE)) %>%
  tibble() %>%
  pmap_dfr(function(estimator, supervision_event, increase) {
    if (supervision_event == "maro_supervision") {
    supervision_events_data  <- generate_treatment(which = "max",
                                             range_years = TREATMENT_YEARS_RANGE,
                                             balance_panel_strategy = "zero",
                                             dataset = "maro_supervision",
                                             flip = !increase,
                                             allow_multiple_treatments = F,
                                             forward_fill_until = max(TREATMENT_YEARS_RANGE),
                                             limit_countries = speeches_countries) %>%
    left_join(speech_agg, by = c("country", "year"))
    } else if (supervision_event == "cbie_policy_q3"){
      supervision_events_data  <- generate_treatment(which = "max",
                                             range_years = TREATMENT_YEARS_RANGE,
                                             balance_panel_strategy = "zero",
                                             dataset = "ro",
                                             flip = increase, # Lower value means more supervision duties
                                             ro_indicator = "cbie_policy_q3",
                                             allow_multiple_treatments = F,
                                             forward_fill_until = max(TREATMENT_YEARS_RANGE),
                                             limit_countries = speeches_countries) %>%
      left_join(speech_agg, by = c("country", "year"))
    }

    if (estimator == "gardner") {
       agg_coef <- gardner_did( "financial_dominance", supervision_events_data, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
      .$estimated_model %>%
      aggregate_lags(supervision_events_data, weight_by_occurnace = T)
    } else if (estimator == "twfe") {
       agg_coef <- twfe_binary("financial_dominance", supervision_events_data, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
      .$estimated_model %>%
      aggregate_lags(supervision_events_data, weight_by_occurnace = T)
    }

    agg_coef %>% mutate(
      supervision_event = supervision_event,
      estimator = estimator,
      increase = increase
    )
  }) %>%
  select(coef, estimator, se, supervision_event, estimator, t_stat, increase) %>%
  mutate(
    coef = case_when(
      abs(t_stat) > 2.58 ~ paste0(format(round(coef, 4), nsmall = 4), "***"),
      abs(t_stat) > 1.96 ~ paste0(format(round(coef, 4), nsmall = 4), "**"),
      abs(t_stat) > 1.64 ~ paste0(format(round(coef, 4), nsmall = 4), "*"),
      TRUE ~ as.character(format(round(coef, 4), nsmall = 4))
    ),
    se = str_glue("({rounded_se})", rounded_se = as.character(format(round(se, 4), nsmall = 4)))
  ) %>%
  pivot_longer(c(coef, se), names_to = "se_or_coef", values_to = "values") %>% select(-t_stat) %>%
  pivot_wider(names_from = estimator, values_from = values) %>%
  select(-se_or_coef) %>%
    mutate(
      supervision_event = case_match(supervision_event,
                                     "maro_supervision" ~ "Masciandaro & Romelli (2018)",
                                        "cbie_policy_q3" ~ "CBI Policy Q3 (Romelli, 2024)"
      )
    ) %>%
    mutate(
      supervision_event = replace(supervision_event, seq_along(supervision_event) %% 2 == 0, ""),
      .by = "supervision_event"
    ) %>%
  select(-increase) %>%
  kbl(
    format = "latex",
    booktabs = T,
    align = c("l", 'c', 'c'),
    col.names = c(" ", "Two-way fixed effects", "Gardner et al. (2024)")
  ) %>%
  column_spec(2:3, width = "4cm") %>%  # Adjusts column width
  pack_rows("Increase", 1, 4) %>%
  pack_rows("Decrease", 5, 8) %>%
  row_spec(2, extra_latex_after = "\\addlinespace[0.2cm]") %>%
  save_kable("./output/tables/did/supervision_events.tex")