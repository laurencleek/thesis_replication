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

# Estimated effect, lagged dependent variable:
lag_structure <- dom[named_dom %in% dominance_to_plot] %>%
  map_dfr(function(dominance) {

    # No lag dep:
   est  <- feols(
      as.formula(str_glue("{dominance} ~ diff_inflation + diff_unemployment  | year + country | ro_cbie_index ~ l1_inverse_distance_weighted_cbi_world + l1_electoral_democracy_10 + independence_judiciary")),
      data = speech_agg_instruments,
      cluster = ~country
    )
   no_lag_dep <- broom::tidy(est) %>%
     mutate(
      dominance = dominance,
      lags = "No lagged DV",
      loglik = logLik(est),
      adjr2 = r2(est)["ar2"]
    )

    past_x_speeches <- c(5,10,25,50,100) %>% map_dfr(
      function(lags) {
      est <- feols(
        as.formula(str_glue("{dominance} ~ l1_dominance + diff_inflation + diff_unemployment  | year + country | ro_cbie_index ~ l1_inverse_distance_weighted_cbi_world + l1_electoral_democracy_10 + independence_judiciary")),
        data = speech_agg_instruments %>%
          arrange(country, date) %>%
          mutate(
            date = as.Date(date),
            l1_dominance = slide_dbl(get(dominance), ~ mean(.x, na.rm = T), .before = lags, .after = -1, .complete = FALSE),
            .by = "country"
          ),
        cluster = ~country
      )

      est %>% broom::tidy() %>% mutate(
        dominance = dominance,
        lags = str_glue(lags, "_speeches"),
        loglik = logLik(est),
        adjr2 = r2(est)["ar2"]
      )
      })

    past_x_days <- c(50,150,300, 500) %>% map_dfr(
      function(lags) {
      est <- feols(
        as.formula(str_glue("{dominance} ~ l1_dominance + diff_inflation + diff_unemployment  | year + country | ro_cbie_index ~ l1_inverse_distance_weighted_cbi_world + l1_electoral_democracy_10 + independence_judiciary")),
        data = speech_agg_instruments %>%
          arrange(country, date) %>%
          mutate(
            date = as.Date(date),
            l1_dominance = slide_index_dbl(get(dominance), date, ~ mean(.x, na.rm = T), .before = lubridate::days(lags), .after = -lubridate::days(1), .complete = FALSE),
            .by = "country"
          ),
        cluster = ~country
      )

      est %>% broom::tidy() %>% mutate(
        dominance = dominance,
        lags = str_glue(lags, "_days"),
        loglik = logLik(est),
        adjr2 = r2(est)["ar2"]
      )
      })

    bind_rows(no_lag_dep, past_x_speeches, past_x_days)
  })

lag_structure %>% filter(term == "fit_ro_cbie_index") %>% mutate(
  estimate = case_when(
      p.value < 0.01 ~ paste0(format(round(estimate, 4), nsmall = 4), "***"),
      p.value < 0.05 ~ paste0(format(round(estimate, 4), nsmall = 4), "**"),
      p.value < 0.1 ~ paste0(format(round(estimate, 4), nsmall = 4), "*"),
      TRUE ~ as.character(format(round(estimate, 4), nsmall = 4))
    ),
  lags = str_replace(lags, "_", " "),
  std.error = str_glue("({rounded_se})", rounded_se = as.character(format(round(std.error, 4), nsmall = 4))),
) %>% select(estimate, std.error, dominance, lags, loglik, adjr2) %>%
  pivot_longer(c(estimate, std.error), names_to = "se_or_coef", values_to = "values") %>%
  mutate(
    loglik = round(mean(loglik), 1),
    adjr2 = as.character(format(round(mean(adjr2), 3), nsmall = 3)),
    .by = "lags") %>%
  pivot_wider(names_from = dominance, values_from = "values") %>%
  select(
    -se_or_coef
  ) %>%
  mutate(
    lags = replace(lags, seq_along(lags) %% 2 == 0, ""),
    loglik = replace(loglik, seq_along(loglik) %% 2 == 0, ""),
    adjr2 = replace(adjr2, seq_along(adjr2) %% 2 == 0, ""),
    .by = "lags"
  ) %>%
  kbl(format = "latex",
      booktabs = T,
      align = c("l", "c", "c", "c", "c"),
      escape = F,
      col.names = c("DV definition:", "Log-Likelihood", "adj. $R^2$", str_replace(dom_label[named_dom %in% dominance_to_plot], "\\s\\w+$", ""))) %>%
  pack_rows(index = c(" " = 2, "Past speeches" = 10, "Calendar days" = 8)) %>%
  add_header_above(c(" " = 1, "Goodness of fit" = 2, "Effect on dominances" = length(dominance_to_plot)))  %>%
  save_kable("./output/tables/iv/lagged_dep_table.tex")