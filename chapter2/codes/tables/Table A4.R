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

ests <- named_dom[named_dom %in% dominance_to_plot] %>% map_dfr(
  function(dom) {

    # Canoncial Diff in Diff:
    canonical_diff_in_diff <- feols(
      as.formula(str_glue("{dom} ~ treated | country + year")),
      data = binary_case,
      cluster = ~country
    )

    canonical_diff_in_diff_agg <- tibble(coef = canonical_diff_in_diff$coefficients, se = canonical_diff_in_diff$se) %>% mutate(
      estimator = "Static difference-in-differences",
      y = dom
    )

    # Equivalent using Gardener
    did2s_simple = did2s(
      binary_case,
      yname = dom, first_stage = ~0 | country + year,
      second_stage = ~treated,
      treatment = "treated",
      cluster_var = "country"
    )

    did2s_simple_agg <- tibble(coef = did2s_simple$coefficients, se = did2s_simple$se) %>% mutate(
      estimator = "Gardener (2022)",
      y = dom
    )

    imput_est <- did_imputation(
      data = binary_case %>% mutate(
        treatment_year = case_when(
          treatment_year == -9999 ~ 0,
          TRUE ~ treatment_year
        )
      ),
      yname = dom,
      gname = "treatment_year",
      tname = "year",
      idname = "country"
    )

    imput_est_agg <- imput_est %>%
      mutate(
        se = std.error,
        coef = estimate,
        estimator = "Borusyak, Jaravel, and Spiess (2024)",
        y = dom
      ) %>%
      select(coef, se, estimator, y) %>%
      tibble()

    callaway_santanna <- cs_did(dom, binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS)
    callaway_santanna_agg <- aggte(callaway_santanna$estimated_model, type = "simple", na.rm = T)
    callaway_santanna_agg <- tibble(
      coef = callaway_santanna_agg$overall.att,
      se = callaway_santanna_agg$overall.se,
      estimator = "Callaway & Sant'Anna (2021)",
      y = dom)

    twfe_agg_equal <- twfe_binary(dom,
                                  binary_case,
                                  leads = EVENT_STUDY_NUMBER_LEADS,
                                  lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
      .$estimated_model %>%
      aggregate_lags() %>%
      select(coef, se) %>%
      mutate(
        estimator = "TWFE aggregation (equal weights)",
        y = dom
      )

    twfe_agg_obs_weighted <- twfe_binary(dom,
                                         binary_case,
                                         leads = EVENT_STUDY_NUMBER_LEADS,
                                         lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
      .$estimated_model %>%
      aggregate_lags(weight_by_occurnace = T, estimation_data = binary_case) %>%
      select(coef, se) %>%
      mutate(
        estimator = "TWFE aggregation (observation weighted)",
        y = dom
      )

    sun_abraham_est <- sun_abraham(y = dom,
                                   data = binary_case,
                                   leads = EVENT_STUDY_NUMBER_LEADS,
                                   lags = EVENT_STUDY_NUMBER_LAGS,
    ) %>%
      .$estimated_model %>%
      summary(agg = "att") %>%
      broom::tidy() %>%
      select(estimate, std.error) %>%
      rename(
        coef = "estimate",
        se = "std.error"
      ) %>%
      mutate(
        estimator = "Sun & Abraham (2021)",
        y = dom
      )

    stacked_did_est_equal <- stacked_did(
      y = dom,
      data = binary_case,
      leads = EVENT_STUDY_NUMBER_LEADS,
      lags = EVENT_STUDY_NUMBER_LAGS
    ) %>%
      .$estimated_model %>%
      aggregate_lags() %>%
      select(coef, se) %>%
      mutate(
        estimator = "Stacked DiD (equal weights)",
        y = dom
      )

    stacked_did_est_weighted <- stacked_did(
      y = dom,
      data = binary_case,
      leads = EVENT_STUDY_NUMBER_LEADS,
      lags = EVENT_STUDY_NUMBER_LAGS
    ) %>%
      .$estimated_model %>%
      aggregate_lags(weight_by_occurnace = T, estimation_data = binary_case) %>%
      select(coef, se) %>%
      mutate(
        estimator = "Stacked DiD (observation weighted)",
        y = dom
      )

    bind_rows(canonical_diff_in_diff_agg,
              twfe_agg_equal,
              twfe_agg_obs_weighted,
              stacked_did_est_equal,
              stacked_did_est_weighted,
              sun_abraham_est,
              callaway_santanna_agg,
              imput_est_agg,
              did2s_simple_agg
    ) })

ests %>%
  mutate(
    t_stat = coef / se,
    coef = case_when(
      abs(t_stat) > 2.58 ~ paste0(format(round(coef, 4), nsmall = 4), "***"),
      abs(t_stat) > 1.96 ~ paste0(format(round(coef, 4), nsmall = 4), "**"),
      abs(t_stat) > 1.64 ~ paste0(format(round(coef, 4), nsmall = 4), "*"),
      TRUE ~ as.character(format(round(coef, 4), nsmall = 4))
    ),
    se = as.character(format(round(se, 4), nsmall = 4))
  ) %>%
  select(coef, estimator, se, y) %>%
  pivot_longer(c(coef, se), names_to = "se_or_coef", values_to = "values") %>%
  mutate(
    values = case_when(
      se_or_coef == "se" ~ paste0("(", values, ")"),
      TRUE ~ values
    )
  ) %>%
  pivot_wider(names_from = y, values_from = "values") %>%
  select(-se_or_coef) %>%
  mutate(
    estimator = replace(estimator, seq_along(estimator) %% 2 == 0, "")
  ) %>%
  kbl(format = "latex", booktabs = T, col.names = c("", str_replace(dom_label[named_dom %in% dominance_to_plot], "\\s\\w+$", ""))) %>%
  add_header_above(c(" " = 1, "Effect on dominances" = length(dominance_to_plot))) %>%
  pack_rows(index = c("Two-way fixed effects" = 10, "Aggregation based" = 4, "Imputation based" = 4)) %>%
  save_kable("./output/tables/did/aggregate_estimate_comp.tex")