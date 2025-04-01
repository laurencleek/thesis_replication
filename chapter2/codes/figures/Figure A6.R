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

## Does CBI affect the audience that you talk to:
# Investigate audiences:
counter <- 1
speeches_prob <- audiences %>% map(function(aud){
  audience_graph <- twfe_binary("audience_lhs",
                            binary_case %>%
                              mutate(audience_lhs = audience == aud),
                                leads = EVENT_STUDY_NUMBER_LEADS,
                                lags = EVENT_STUDY_NUMBER_LAGS,
                                fill = EVENT_STUDY_FILL) %>%
    plot_estimates(manual_breaks = c(-5, -3, -1, 0, 2, 4, 6, 8, 10, 12))

  audience_graph <- audience_graph + ggtitle(str_glue(
    "{letter}. {audience_graph}",
    letter = LETTERS[counter],
    audience_graph = names(named_audiences)[audiences == aud]
  )) +
    theme(
      axis.title.y = element_text(margin = margin(t = 0, r = 2, b = 0, l = 0, "pt")), # Fix spacing between charts
      plot.margin = margin(5, 5, 5, 5, "pt")
    )

  counter <<- counter + 1

  audience_graph
})

wrap_plots(speeches_prob, ncol = 2) %>%
   ggsave(
    filename = "output/figures/did/effect_on_audience.pdf",
    width = panel_chart_width,
    height = panel_chart_height * 0.7,
    device = cairo_pdf,
    units = "cm"
  )