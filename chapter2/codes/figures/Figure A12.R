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

run_placebo_simulations <- FALSE # If set to false attempt to load from ./data/processed/placebo_simulations folder
# Simulate random treatments:
set.seed(123)
generate_seeds <- sample(1000000, size = 10000, replace = F)

if (run_placebo_simulations) {
  placebo_estimates_full_random <- pmap_dfr(expand.grid(seed = generate_seeds, dom = dom), function(seed, dom) {
    randomized_treatment <- binary_case_treatments %>%
      filter(year <= 2023) %>% # Do not allow treatment to be post 2023
      randomize_treatments(randomize_seed = seed) %>%
      left_join(speech_agg, by = c("country", "year"))

    twfe_estimation <- twfe_binary(dom, randomized_treatment, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL)

    twfe_estimation$estimated_model %>%
      aggregate_lags(weight_by_occurnace = T, estimation_data = randomized_treatment) %>%
      mutate(
        seed = seed,
        type = "agg",
        beta = coef
      ) %>%
      select(beta, se, type, seed) %>%
      bind_rows(
        twfe_estimation$estimates
      ) %>%
      mutate(
        seed = seed,
        dominance = dom
      )
  })
  write_parquet(placebo_estimates_full_random, "./data/processed/placebo_simulations/placebo_estimates_full_random.parquet")
} else {
  placebo_estimates_full_random <- read_parquet("./data/processed/placebo_simulations/placebo_estimates_full_random.parquet")
}

inidividual_randomize_placebo_plot_beta <- placebo_estimates_full_random %>%
  filter(dominance %in% dominance_to_plot) %>%
  filter(type == "es_coef") %>%
  mutate(event_time = factor(event_time, levels = rev(unique(event_time)))) %>%
  mutate(
    dominance = ordered(dominance,
                        levels = named_dom,
                        labels = names(named_dom))
  ) %>%
  ggplot(aes(x = beta, y = event_time)) +
  geom_density_ridges(linewidth = 0.35, quantile_lines = TRUE, scale = 0.9, color = colors[1], fill = colors[1], alpha = 0.5, quantile_fun = function(beta, ...)mean(beta)) +
  paper_theme() +
  facet_wrap(~dominance, scales = "free") +
  labs(
    x = "",
    y = "Time to independence increase",
  ) +
  paper_theme() +
  scale_x_continuous(n.breaks = 3) +
  ggtitle("A. Event study coefficient")

inidividual_randomize_placebo_plot_se <- placebo_estimates_full_random %>%
  filter(dominance %in% dominance_to_plot) %>%
  filter(type == "es_coef") %>%
  mutate(event_time = factor(event_time, levels = rev(unique(event_time)))) %>%
  mutate(
    dominance = ordered(dominance,
                        levels = named_dom,
                        labels = names(named_dom))
  ) %>%
  mutate(
    t_stat = beta / se
  ) %>%
  ggplot(aes(x = t_stat, y = event_time)) +
  geom_density_ridges(linewidth = 0.35, quantile_lines = TRUE, scale = 0.9, color = colors[1], fill = colors[1], alpha = 0.5, quantile_fun = function(beta, ...)mean(beta)) +
  paper_theme() +
  facet_wrap(~dominance, scales = "free") +
  labs(
    x = "",
    y = "Time to independence increase",
  ) +
  paper_theme() +
  scale_x_continuous(limits = c(-4, 4)) +
  ggtitle("B. t-statistic")

ggsave(
  inidividual_randomize_placebo_plot_beta +
    inidividual_randomize_placebo_plot_se +
    plot_layout(ncol = 1),
  filename = "output/figures/did/placebo_individual.pdf",
  width = ifelse(export_to_slides, panel_slide_width, default_chart_width),
  height = ifelse(export_to_slides, panel_slide_height, panel_chart_height + 2),
  device = cairo_pdf,
  units = "cm"
)