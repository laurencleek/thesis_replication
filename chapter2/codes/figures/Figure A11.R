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

# Need this counter variable here to label that as A, B, C
count <- 1
aggregate_randomize_placebo_plot <- named_dom[named_dom %in% dominance_to_plot] %>%
  imap(
    function(dom, name) {
      # event_study_estimate:
      event_stufy_estimate <- twfe_binary(dom, binary_case, leads = EVENT_STUDY_NUMBER_LEADS, lags = EVENT_STUDY_NUMBER_LAGS, fill = EVENT_STUDY_FILL) %>%
        .$estimated_model %>%
        aggregate_lags(weight_by_occurnace = T, estimation_data = binary_case) %>%
        pull(coef)
      mean_agg_estimate <- placebo_estimates_full_random %>%
        filter(type == "agg", dominance == dom) %>%
        pull(beta) %>%
        mean()

      # Limits
      limit <- list(
        monetary_dominance = 0.2,
        fiscal_dominance = 0.03,
        financial_dominance = 0.06
      )[[dom]]

      y_label <- list(
        monetary_dominance = 7,
        fiscal_dominance = 47,
        financial_dominance = 23
      )[[dom]]

      h_just_random <- list(
        monetary_dominance = -0.05,
        fiscal_dominance = 1.05,
        financial_dominance = 1.05
      )[[dom]]

      h_just_es <- list(
        monetary_dominance = -0.05,
        fiscal_dominance = -0.05,
        financial_dominance = 1.05
      )[[dom]]

      all_placebo_ests <- placebo_estimates_full_random %>%
        filter(type == "agg", dominance == dom) %>%
        pull(beta) %>%
        ecdf()

      print(str_glue("Location in distribution of estiamte is {percent}", percent = all_placebo_ests(event_stufy_estimate)))

      plot <- placebo_estimates_full_random %>%
        filter(type == "agg", dominance == dom) %>%
        ggplot(aes(beta)) +
        geom_histogram(aes(y = ..density..), color = colors[1], fill = colors[1], alpha = HIST_FILL_ALPHA, breaks = seq(-limit, limit, 2 * limit / 40)) +
        geom_density(linetype = "dashed", color = colors[1]) +
        geom_vline(color = colors[1], xintercept = mean_agg_estimate, linetype = "dashed") +
        geom_vline(color = colors[1], xintercept = event_stufy_estimate, linetype = "dashed") +
        geom_text(aes(x = event_stufy_estimate, label = str_glue("ES estimate = {round(event_stufy_estimate,3)}\n({round(all_placebo_ests(event_stufy_estimate) * 100, 2)} percentile)"), y = y_label), check_overlap = TRUE, vjust = 1, , hjust = h_just_es, size = 2) +
        geom_text(aes(x = mean_agg_estimate, label = str_glue("Randomization mean = {round(mean_agg_estimate,5)}"), y = y_label), check_overlap = TRUE, vjust = 1, hjust = h_just_random, size = 2) +
        scale_y_continuous(expand = expansion(mult = c(0, .04))) +
        scale_x_continuous(limits = c(-limit, limit)) +
        labs(
          x = "Aggregated Coefficient",
          y = "Density"
        ) +
        ggtitle(str_glue("{toupper(letters[count])}. {name}")) +
        paper_theme()

      count <<- count + 1
      print(count)
      return(plot)
    }
  ) %>%
  patchwork::wrap_plots(ncol = if (export_to_slides) 2 else 1,
                        axes_titles = "axis_titles",
                        design = if (export_to_slides) patchwork2x1_design else NULL)

ggsave(
  aggregate_randomize_placebo_plot,
  filename = "output/figures/did/placebo_aggregate.pdf",
  width = ifelse(export_to_slides, panel_slide_width, default_chart_width),
  height = ifelse(export_to_slides, panel_slide_height, (2/3) * default_chart_height * length(dominance_to_plot)),
  device = cairo_pdf,
  units = "cm"
)