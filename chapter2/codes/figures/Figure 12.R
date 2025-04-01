library(tidyverse)
library(readxl)
library(haven)
library(arrow)

source("./codes/functions/chart_themes.R")
source("./codes/functions/diff_in_diff_functions.R")
source("./codes/constants/diff_in_diff_settings.R")

# Load data
speech_agg <- read_parquet("./data/processed/aggregations/speech_agg.parquet") %>% filter(
  year >= 1997,
  year <= 2023
)

speeches_countries <- setdiff(speech_agg %>% pull(country) %>% unique(), "")

# Binary case:
binary_case_treatments <- generate_treatment(which = "max",
                                             minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                                             range_years = TREATMENT_YEARS_RANGE,
                                             balance_panel_strategy = "zero",
                                             dataset = "ro",
                                             forward_fill_until = max(TREATMENT_YEARS_RANGE),
                                             allow_multiple_treatments = F,
                                             limit_countries = speeches_countries)

binary_case <- left_join(binary_case_treatments, speech_agg, by = c("country", "year"))

#' Aggregate coefficients with custom weights
#'
#' This function aggregates coefficient estimates using custom weights and calculates
#' the corresponding standard errors for the aggregated coefficients.
#'
#' @param coef Numeric vector. Vector of all coefficient estimates to be aggregated.
#' @param vcov Matrix. Variance-covariance matrix of all estimates.
#' @param weights Matrix. Weight matrix where each row corresponds to a desired aggregation.
#'                Rows will be normalized to sum to 1.
#'
#' @return A tibble with two columns:
#'   \item{coef}{Numeric. The aggregated coefficient estimates.}
#'   \item{std.error}{Numeric. The standard errors of the aggregated estimates.}
aggregate_coefficients <- function(coef, # Vector of all coefficient estimates
                                   vcov, # Covariance matrix of all estimates
                                   weights # Weight matrix (one row for each aggregation)
) {
  # Ensure that weights are summing to one:
  weights <- apply(weights, 1, function(row) row / sum(row)) %>% t()

  tibble(
    coef = as.vector(weights %*% coef),
    std.error = sqrt(diag(weights %*% vcov %*% t(weights)))
  )
}

# Estimate Sun & Abraham Event study without binning coefficients
sunab <- sun_abraham(y = "financial_dominance", data = binary_case)

# Extract all relevant estimates and calculate the year of the coefficient
subab_coefficients_with_year_cohort_information <- summary(sunab$estimated_model, agg = FALSE) %>%
  tidy() %>%
  mutate(cohort = as.numeric(str_extract(term, "(?<=cohort::)\\d+")), time_post_treat = as.numeric(str_extract(term, "(?<=time_post_treat::)-?\\d+"))) %>%
  mutate(year = cohort + time_post_treat) %>%
  left_join(
    count_observations_by_cohort_year(binary_case), by = join_by(year == year, cohort == treatment_year)
  )

# Calculate weights for aggregations:
year_aggregation <- subab_coefficients_with_year_cohort_information %>%
  mutate(weight = (year >= cohort & year >= 1997) * count) %>%  # We want to aggregate post coefficinets. Anything with year < cohort is pre treatment and should have weight 0
  pivot_wider(names_from = year, values_from = weight, values_fill = 0) %>%
  select(matches("^\\d+$"))

# Define pre and post crisis period and set weight of coefficients that are pre-treatment to zero
crisis_aggregation <- subab_coefficients_with_year_cohort_information %>%
  mutate(
    weight = (year >= cohort) * count,
    period = case_when(
      year < 2009 ~ "pre_crisis",
      year >= 2009 ~ "post_crisis"
    )
  ) %>%  # We want to aggregate post coefficinets. Anything with year < cohort is pre treatment and should have weight 0
  pivot_wider(names_from = period, values_from = weight, values_fill = 0) %>%
  select(matches("_crisis$"))

# Aggregate into single number. Should produce the same as summary(sunab$estimated_model, agg = "att") and it does!
att <- subab_coefficients_with_year_cohort_information %>%
  mutate(weight = (year >= cohort) * count) %>%
  select(weight)

att_agg <- aggregate_coefficients(summary(sunab$estimated_model, agg = FALSE) %>% coef(), vcov(sunab$estimated_model), att %>% t())

# Calculate pre-crisis and post-crisis averages
crisis_aggregation_agg <- aggregate_coefficients(summary(sunab$estimated_model, agg = FALSE) %>% coef(), vcov(sunab$estimated_model), crisis_aggregation %>% t()) %>%
  mutate(
    pre_post = names(crisis_aggregation)
  )

# Calculate year averages
aggregate_coefficients(summary(sunab$estimated_model, agg = FALSE) %>% coef(), vcov(sunab$estimated_model), year_aggregation %>% t()) %>%
  mutate(
    year = names(year_aggregation)
  ) %>%
  filter(
    year >= 1998
  ) %>%
  mutate(
    pre_crisis_coef = as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == "pre_crisis", "coef"]),
    pre_crisis_se = as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == "pre_crisis", "std.error"]),
    post_crisis_coef = as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == "post_crisis", "coef"]),
    post_crisis_se = as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == "post_crisis", "std.error"])
  ) %>%
  mutate(
    pre_crisis_coef = ifelse(year < 2009, pre_crisis_coef, NA),
    pre_crisis_se = ifelse(year < 2009, pre_crisis_se, NA),
    post_crisis_coef = ifelse(year >= 2009, post_crisis_coef, NA),
    post_crisis_se = ifelse(year >= 2009, post_crisis_se, NA)
  ) %>%
  mutate(
    period = case_when(
      year < 2009 ~ "Before global financial crisis",
      year >= 2009 ~ "After global financial crisis"
    )
  ) %>%
  mutate(
    year = as.numeric(year)
  ) %>%
  ggplot(
    aes(x = year, y = coef, color = period, shape = period, fill = period)
  ) +
  geom_hline(yintercept = 0, linewidth = 0.35) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = coef - 1.96 * std.error, ymax = coef + 1.96 * std.error), width = 0.35, linewidth = 0.35) +
  paper_theme() +
  geom_line(aes(y = pre_crisis_coef, x = year, group = 1), alpha = 0.6, linetype = "dashed", linewidth = 0.35) +
  geom_line(aes(y = post_crisis_coef, x = year, group = 1), alpha = 0.6, linetype = "dashed", linewidth = 0.35) +
  geom_ribbon(aes(ymin = post_crisis_coef + 1.96 * post_crisis_se, ymax = post_crisis_coef - 1.96 * post_crisis_se, group = 1), fill = colors[1], color = NA, alpha = 0.2) +
  geom_ribbon(aes(ymin = pre_crisis_coef + 1.96 * pre_crisis_se, ymax = pre_crisis_coef - 1.96 * pre_crisis_se, group = 1), fill = colors[2], color = NA, alpha = 0.2) +
  ylim(c(-0.1, 0.125)) +
  geom_text(aes(x = (1998 + 2008) / 2, label =
    str_glue("ATT before GFC = {format_number(as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == 'pre_crisis', 'coef']), 3)} ({format_number(as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == 'pre_crisis', 'std.error']), 3)})"),
                y = 0.109), color = colors[2], hjust = 0.5, size = 2.5, check_overlap = TRUE, family = "XCharter Math") +
  geom_text(aes(x = (2009 + 2023) / 2, label = str_glue("ATT after GFC = {format_number(as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == 'post_crisis', 'coef']), 3)} ({format_number(as.numeric(crisis_aggregation_agg[crisis_aggregation_agg$pre_post == 'post_crisis', 'std.error']),3 )})"),
                y = 0.109), color = colors[1], hjust = 0.5, size = 2.5, check_overlap = TRUE, family = "XCharter Math") +
  labs(
    y = "Average Treatment Effect on the Treated (ATT)",
    x = "Year",
    color = "",
    shape = "",
    fill = ""
  ) +
  scale_y_continuous(labels = scales::percent, limits = c(-0.1, 0.115)) +
  discrete_scale('shape', 'shape_d', function(n) c(21, 23, 22)[seq_len(n)]) +
  theme(
    axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 1))
  )

ggsave(
  filename = "output/figures/did/calendar_time_aggregation.pdf",
  width = default_chart_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)