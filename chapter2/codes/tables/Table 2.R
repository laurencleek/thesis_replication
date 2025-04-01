library(tidyverse)
library(readxl)
library(kableExtra)
library(arrow)

# Load Chart themes
source("./codes/functions/chart_themes.R")

# Load functions and settings
source("./codes/functions/diff_in_diff_functions.R")
source("./codes/constants/diff_in_diff_settings.R")
source("./codes/constants/varnames.R")

speech_agg <- read_parquet("./data/processed/aggregations/speech_agg.parquet") %>% filter(
  year >= 1997,
  year <= 2023
)

# These are the actual countries we have in the speeches dataset
speeches_countries <- setdiff(speech_agg %>% pull(country) %>% unique(), "")

binary_case <- generate_treatment(which = "max",
                                  minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                  range_years = TREATMENT_YEARS_RANGE,
                                  balance_panel_strategy = "zero",
                                  limit_countries = speeches_countries,
                                  allow_multiple_treatments = F) %>%
  left_join(speech_agg, by = c("country", "year"))

# Heterogeneity. Loop over dominances and calculate multiple dimensions of heterogeneity
het <- named_dom[named_dom %in% dominance_to_plot] %>% map_dfr(
  function(dom) {
    baseline <- twfe_binary(dom,
                            binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
      .$estimated_model %>%
      aggregate_lags(estimation_data = binary_case, weight_by_occurnace = T) %>%
      mutate(
        labels = c("Full sample"),
        dominance = dom,
        heterogeneity_dimension = "Baseline"
      )

    advanced <- twfe_binary(dom,
                            binary_case %>% mutate(advanced = factor(advanced)),
                            interact = "advanced", leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
      .$estimated_model %>%
      aggregate_lags(estimation_data = binary_case, weight_by_occurnace = T) %>%
      mutate(
        labels = c("Emerging and Developing", "Advanced"),
        dominance = dom,
        heterogeneity_dimension = "Economic development"
      )

    democracy <- twfe_binary(dom,
                             binary_case %>% mutate(democracy_ind = factor(democracy_ind)),
                             interact = "democracy_ind", leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
      .$estimated_model %>%
      aggregate_lags(estimation_data = binary_case, weight_by_occurnace = T) %>%
      mutate(
        labels = c("Autocracy", "Democracy"),
        dominance = dom,
        heterogeneity_dimension = "Political system"
      )

    supervision <- twfe_binary(dom,
                               binary_case %>%
                                 mutate(
                                   supervision_capability = case_when(
                                     cbis %in% c(1, 2) ~ "low",
                                     cbis %in% c(3) ~ "medium",
                                     cbis %in% c(4, 5, 6) ~ "high",
                                     TRUE ~ NA
                                   ),
                                   supervision_capability = factor(supervision_capability, levels = c("low", "medium", "high"))
                                 ) %>%
                                 group_by(country) %>%
                                 fill(supervision_capability, .direction = "down") %>%
                                 ungroup(),
                               interact = "supervision_capability", leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL
    ) %>%
      .$estimated_model %>%
      aggregate_lags(estimation_data = binary_case, weight_by_occurnace = T) %>%
      mutate(
        labels = c("Low", "Medium", "High"),
        dominance = dom,
        heterogeneity_dimension = "Supervision capabilities"
      )

    own_monetary_policy <- twfe_binary(dom,
                                       binary_case %>%
                                         mutate(
                                           pegged_or_currency_union = factor(member_of_currency_union | pegged)
                                         ) %>%
                                         group_by(country) %>%
                                         ungroup(),
                                       interact = "pegged_or_currency_union", leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL
    ) %>%
      .$estimated_model %>%
      aggregate_lags(estimation_data = binary_case, weight_by_occurnace = T) %>%
      mutate(
        labels = c("Full monetary sovereignty", "Monetary union or peg"),
        dominance = dom,
        heterogeneity_dimension = "Monetary sovereignty"
      )

    mandates <- twfe_binary(dom,
                            binary_case %>%
                              mutate(
                                mandate = case_when(
                                                    ro_cbie_obj >= 0.5 ~ "Non-conflicting with price stability",
                                                    ro_cbie_obj <= 0.25 ~ "Conflicting objectives",
                                                    TRUE ~ NA
                                ),
                                mandate = factor(mandate, levels = c("Non-conflicting with price stability", "Conflicting objectives"))
                              ),
                                interact = "mandate", leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL
    ) %>%
      .$estimated_model %>%
      aggregate_lags(estimation_data = binary_case, weight_by_occurnace = T) %>%
      mutate(
        labels = c("Non-conflicting with price stability", "Conflicting objectives"),
        dominance = dom,
        heterogeneity_dimension = "Mandates"
      )

    bind_rows(
      baseline,
      supervision,
      democracy,
      own_monetary_policy,
      advanced,
      mandates
    )
  }
)

reorder_by_hetgen_dim <- het %>%
  mutate(
    coef = case_when(
      abs(t_stat) > 2.58 ~ paste0(format(round(coef, 4), nsmall = 4), "***"),
      abs(t_stat) > 1.96 ~ paste0(format(round(coef, 4), nsmall = 4), "**"),
      abs(t_stat) > 1.64 ~ paste0(format(round(coef, 4), nsmall = 4), "*"),
      TRUE ~ as.character(format(round(coef, 4), nsmall = 4))
    ),
    se = str_glue("({rounded_se})", rounded_se = as.character(format(round(se, 4), nsmall = 4)))
  ) %>%
  select(heterogeneity_dimension, dominance, labels, se, coef) %>%
  pivot_longer(c(coef, se), names_to = "se_or_coef", values_to = "values") %>%
  pivot_wider(names_from = dominance, values_from = "values")

# Export table
reorder_by_hetgen_dim %>%
  select(
    -heterogeneity_dimension, -se_or_coef
  ) %>%
  mutate(
    labels = replace(labels, seq_along(labels) %% 2 == 0, ""),
    .by = "labels"
  ) %>%
  kbl(format = "latex", booktabs = T, align = c("l", rep('c', length(dominance_to_plot))), col.names = c(" ", names(named_dom[named_dom %in% dominance_to_plot]))) %>%
  pack_rows(index = table(fct_inorder(reorder_by_hetgen_dim$heterogeneity_dimension))) %>%
  save_kable("./output/tables/did/effect_heterogeneity.tex")