library(geosphere)
library(tidyverse)
library(haven)
library(arrow)
library(fixest)
library(readxl)
library(tictoc)
library(slider)

# This is an attempt of running a diff in diff or FE regression that instruments CBI with inverted distances
source("./codes/constants/varnames.R")

# Chart themes
source("./codes/functions/chart_themes.R")

# Run instrument calculation
run_instrument_calculation <- FALSE # Set to true to calculate instruments from VDEM and metadata.

# Load speech aggregation
speech_agg <- read_parquet("./data/processed/aggregations/speech_agg.parquet") %>% filter(
    year >= 1997,
    year <= 2023
)

if (run_instrument_calculation) {
  # Load the Romelli Dataset:
  independence_romelli <- read_excel("./data/input/metadata/central_bank/CBIData_Romelli_2024.xlsx", sheet = "CBI data") %>%
    select(iso_a3, year, all_of((c("cbie_index", "cbie_board", "cbie_policy", "cbie_obj", "cbie_lending", "cbie_finindep", "cbie_report"))))

  # Load vdem dataset
  vdem_democracy <- read_parquet("./data/processed/vdem_clean.parquet") %>%
    mutate(
      democracy_ind = case_when(
        v2x_regime == 0 ~ 0,
        v2x_regime == 1 ~ 0,
        v2x_regime == 2 ~ 1,
        v2x_regime == 3 ~ 1,
        TRUE ~ NA)) %>%
    select(country, year, v2x_libdem, v2x_polyarchy, democracy_ind)

  # Merge central bank locations with CBI data
  country_locs_and_insts <- independence_romelli %>%
    left_join(
      read.csv("./data/input/metadata/central_bank/currencies_country.csv") %>%
        select(country, cb_longitude, cb_latitude) %>%
        summarise(
          across(everything(), ~mean(.x)),
          .by = "country"
        ),
      by = join_by(iso_a3 == country)
    ) %>%
    rename(
      country = "iso_a3"
    ) %>%
    filter(
      !is.na(cb_longitude) & !is.na(cb_latitude)
    ) %>%
    left_join(
      vdem_democracy,
      by = join_by(country == country, year == year)
    ) %>%
    select(country, year, cb_longitude, cb_latitude, cbie_index, v2x_libdem, v2x_polyarchy, democracy_ind)

  distance_to <- function(long, lat, long_ref, lat_ref) {
    data.frame(long = long, lat = lat, long_ref = long_ref, lat_ref = lat_ref) %>%
      rowwise() %>%
      mutate(
        dist = distm(c(long, lat), c(long_ref, lat_ref), fun = distHaversine)[1, 1] / 1000
      ) %>%
      pull(dist)
  }

  # This function is terribly slow
  calculate_inv_dist_weights <- function(data,
                                         variable,
                                         for_which_year = 2015,
                                         for_which_country = "USA",
                                         beta = 1, top_x = NULL,
                                         equal_weights = FALSE
  ) {
    ref_long <- data %>%
      filter(country == for_which_country, year == for_which_year) %>%
      select(cb_longitude) %>%
      pull()
    ref_lat <- data %>%
      filter(country == for_which_country, year == for_which_year) %>%
      select(cb_latitude) %>%
      pull()

    weights <- data %>%
      filter(for_which_year == year, !is.na(get(variable)))

    # Return NA in case there are no
    if (nrow(weights) == 0) {
      return(NA)
    }

    weights <- weights %>%
      mutate(
        weight = 1 / (distance_to(cb_longitude, cb_latitude, ref_long, ref_lat)^beta)
      ) %>%
      mutate(
        weight = ifelse(weight == Inf, 0, weight)
      ) %>%
      mutate(
        weight = weight / sum(weight) # Make weights sum to 1
      )

    if (!is.null(top_x)) {
      weights <- weights %>%
        arrange(-weight) %>%
        head(top_x) %>%
        mutate(
          weight = weight / sum(weight) # Need to normalize them to sum again to one again
        )
    }

    if (equal_weights) {
      weights <- weights %>%
        mutate(
          weight = 1 / nrow(weights)
        )
    }

    return(sum(weights[[variable]] * weights$weight))
  }

  # Calculate set of possible instruments
  instruments <- expand_grid(equal_weighted = c(TRUE, FALSE), top_x = c(5, 10, 25, NA), instrument = c("cbi", "electoral_democracy", "liberal_democracy", "binary_democracy")) %>%
    pmap_dfc(function(equal_weighted, top_x, instrument) {
      country_locs_and_insts %>%
        select(year, country) %>%
        pmap_dbl(
          function(year, country) {
            calculate_inv_dist_weights(country_locs_and_insts, case_match(instrument, "cbi" ~ "cbie_index",
                                                                          "electoral_democracy" ~ "v2x_polyarchy",
                                                                          "binary_democracy" ~ "democracy_ind",
                                                                          "liberal_democracy" ~ "v2x_libdem"),
                                       year,
                                       country,
                                       equal_weights = equal_weighted,
                                       top_x = if (is.na(top_x)) NULL else top_x,
                                       beta = 1)
          }
        ) %>%
        tibble() %>%
        set_names(str_glue("{ifelse(!equal_weighted, 'inverse_distance_weighted_', '')}{instrument}_{ifelse(is.na(top_x),'world' , top_x)}"))
    })

  distance_weighted_instruments <- country_locs_and_insts %>%
    select(country, year) %>%
    bind_cols(instruments %>% mutate(across(everything(), ~lag(.x), .names = "l1_{.col}")))

  write_parquet(distance_weighted_instruments, "./data/processed/iv_instruments.parquet")

} else {
  distance_weighted_instruments <- read_parquet("./data/processed/iv_instruments.parquet")
}

# Create dataset with instruments joined
speech_agg_instruments <- speech_agg %>% left_join(
  distance_weighted_instruments,
  by = join_by(year == year, country == country)
) %>% mutate(
  # Create a sort of lagged dependent variables based on the last 25 speeches
  l1_financial_dominance = slide_dbl(financial_dominance, ~ mean(.x, na.rm = T), .before = 25, .after = -1, .complete = FALSE),
  l1_monetary_dominance = slide_dbl(monetary_dominance, ~ mean(.x, na.rm = T), .before = 25, .after = -1, .complete = FALSE),
  .by = "country"
)

# Then for interaced versions
unnest_list_by_one_level <- function(list_of_lists) {
  new_list <- list()
  for (lists in list_of_lists) {
    new_list <- c(new_list, lists)
  }
  return(new_list)
}

counter <- 1
c("advanced", "democracy_ind") %>%
  map(function(interact) {
    dom[named_dom %in% dominance_to_plot] %>% map(function(dominance) {

      two_stage_ls <- feols(
        as.formula(str_glue("{dominance} ~ l1_{dominance} + diff_inflation + diff_unemployment  | country + year | ro_cbie_index + ro_cbie_index:{interact} ~ l1_inverse_distance_weighted_cbi_world + l1_electoral_democracy_10 + independence_judiciary + l1_inverse_distance_weighted_cbi_world:{interact} + l1_electoral_democracy_10:{interact} + independence_judiciary:{interact}")),
        data = speech_agg_instruments,
        cluster = ~country
      )

      if (FALSE) {
        counter <<- counter + 1
        return(summary(two_stage_ls, stage = 1:2))
      } else {
        return(summary(two_stage_ls, stage = 2))
      }
    })
  }) %>%
  unnest_list_by_one_level() %>%
  etable(depvar = T,
         powerBelow = -6,
         dict = c("ro_cbie_index" = "$\\widehat{\\textrm{CBI}}$",
                  "monetary_dominance" = "Monetary",
                  "fiscal_dominance" = "Fiscal",
                  "financial_dominance" = "Financial",
                  "country" = "Country",
                  "year" = "Year",
                  "l1_monetary_dominance" = "Lagged DV", # Should this be separated?
                  "l1_financial_dominance" = "Lagged DV",
                  "advanced" = "Advanced economy",
                  "democracy_ind" = "Democracy",
                  "ro_cbie_index:advanced" = "$\\widehat{\\textrm{CBI} \\times \\textrm{Advanced}}$",
                  "ro_cbie_index:democracy_ind" = "$\\widehat{\\textrm{CBI} \\times \\textrm{Democracy}}$",
                  "l1_inverse_distance_weighted_cbi_world" = "Inverse distance weighted world CBI\\textsubscript{-1}",
                  "l1_electoral_democracy_10" = "Neighbour's electoral democracy index\\textsubscript{-1}",
                  "independence_judiciary" = "Independence judiciary",
                  "diff_unemployment" = "$\\Delta$Unemployment rate",
                  "diff_inflation" = "$\\Delta$Inflation rate",
                  "democracy_ind" = "Democracy"),
         order = c("\\$\\\\widehat\\{\\\\textrm\\{CBI\\}"),
         tex = TRUE,
         fitstat = ~r2 + n,
         style.tex = style.tex("base",
                               tablefoot = FALSE,
                               fixef.title = "\\midrule \\emph{Fixed Effects}",
                               yesNo = "$\\checkmark$"),
         headers = list("^:_:Interaction" = list("Advanced" = length(dominance_to_plot), "Democracy" = length(dominance_to_plot)))) %>%
  write_lines("./output/tables/iv/iv_country_fe_interacted.tex")