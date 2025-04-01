#DiD Libraries
library(did)
library(did2s)
library(fixest)
library(countrycode)
library(didimputation)
library(haven)



# Define a cache here for the Excel file. This speeds up the "bootstrap" like placebo tests that run the estimation many times
# with randmized treatments
cache <- list()

#' Generate Treatment Data for Central Bank Independence Analysis
#'
#' This function processes central bank independence data to generate treatment
#' variables for difference-in-differences or event study analyses. It can handle
#' data from multiple sources and offers various options for treatment definition
#' and intensity calculation.
#'
#' @param dataset Character. Dataset to use. Options: "ro" (Romelli), "ga" (Garriga).
#' @param range_years Numeric vector. Years to consider for independence changes.
#' @param flip Logical. If TRUE, flips treatment intensity (for studying decreases).
#' @param minimum_intensity Numeric. Minimum absolute value to consider as treatment.
#' @param negative_strategy Character. How to handle negative values: "allow", "drop", or "zero".
#' @param top_x Numeric or NULL. If set, only selects x highest treatments by intensity.
#' @param allow_multiple_treatments Logical. If TRUE, allows multiple treatments per country.
#' @param which Character. Method to calculate time_post_treat and select among multiple treatments: "drop", "min", "max", or "closest"
#' @param balance_panel_strategy Character. Strategy for panel balancing: "unbalanced", "zero", or "missing".
#' @param cumulate_intensity Character or NULL. If set, cumulates intensity values. Only makes sense with allow_multiple_treatments
#' @param never_treat_treatment_year Numeric. Treatment year assigned to never-treated units. Callaway Sant Anna expects 0.
#' @param never_treat_time_post_treat Numeric. Time post-treatment for never-treated units. Relevant for TWFE if estiamted using i() and ref category
#' @param forward_fill_until Numeric or NULL. Extend dataset into future with no treatments.
#' @param limit_countries Character vector or NULL. Limit analysis to specific countries.
#' @param offset_treatmnet Numeric. Option to shift treatment timing.
#' @param ro_indicator Character. Indicator to use for Romelli dataset.
#' @param ro_add_ecb_to_control Logical. Whether to add ECB to control group for Romelli dataset.
#' @param ga_indicator Character. Indicator to use for Garriga dataset.
#' @param ga_add_ecb_to_control Logical. Whether to add ECB to control group for Garriga dataset.
#'
#' @return A data frame with treatment variables for each country-year observation.
generate_treatment <- function(
  dataset = "ro",
  range_years = 1997:2023,
  flip = FALSE,
  minimum_intensity = 0,
  negative_strategy = "zero",
  top_x = NULL,
  allow_multiple_treatments = FALSE,
  which = "max",
  balance_panel_strategy = "unbalanced",
  cumulate_intensity = NULL,
  never_treat_treatment_year = 0,
  never_treat_time_post_treat = -9999,
  forward_fill_until = NULL,
  limit_countries = NULL,
  offset_treatmnet = 0,
  # Dataset specific options
  # Romelli
  ro_indicator = "cbie_index",
  ro_add_ecb_to_control = FALSE,
  # Garriga
  ga_indicator = "lvau_garriga",
  ga_add_ecb_to_control = FALSE
) {

  # Return a vector of the closest year
  find_closest_year <- function(years, intensity, resolve_equal_distance = "earlier") {
    if (isTRUE(any(intensity != 0))) {
      return(
        sapply(years, function(x) {
          event <- years[intensity != 0]
          event <- na.omit(event)

          closest_match <- event[abs(event - x) == min(abs(event - x))]
          if (length(closest_match) == 2) {
            if (resolve_equal_distance == "earlier") {
              closest_match <- closest_match[1]
            } else {
              closest_match <- closest_match[2]
            }
          }
          return(closest_match)
        }))
    } else {
      return(rep(never_treat_treatment_year, length(years)))
    }
  }

  # Retruns a single year
  find_treatment_year <- function(year, intensity, how = "max", resolve_equal_intensity = "earlier") {
    if (isTRUE(any(intensity != 0))) {
      if (how == "max") {
        max_treatment_index <- which(abs(intensity) == max(abs(intensity), na.rm = T))
        if (length(max_treatment_index) > 1) {
          warning("Max intensity not unique")
          if (resolve_equal_intensity == "earlier") {
            max_treatment_index <- max_treatment_index[1]
          } else if (resolve_equal_intensity == "later") {
            max_treatment_index <- max_treatment_index[2]
          }
        }
        return(year[max_treatment_index])
      } else if (how == "first" | how == "last") {
        any_treatment_index <- which(intensity != 0)
        if (how == "first") {
          return(year[any_treatment_index[1]])
        } else if (how == "last") {
          return(year[any_treatment_index[length(any_treatment_index)]])
        }
      }
    } else {
      return(never_treat_treatment_year)
    } }

  fill_upwards <- function(x) {
    first_non_na <- x[!is.na(x)][1]
    nas <- 1:sum(cumprod(is.na(x)))
    x[nas] <- first_non_na
    return(x)
  }

  if (dataset == "ro") {
    if ("CBIData_Romelli_2024.xlsx" %in% names(cache)) {
      data <- cache[["CBIData_Romelli_2024.xlsx"]]
    } else {
      data <- read_excel("./data/input/metadata/central_bank/CBIData_Romelli_2024.xlsx", sheet = "CBI data")
      cache[["CBIData_Romelli_2024.xlsx"]] <<- data
    }
    data <- data %>%
      select(iso_a3, year, all_of((c("cbie_index", "cbie_board", "cbie_policy", "cbie_obj", "cbie_lending", "cbie_finindep", "cbie_report", "cbie_policy_q3")))) %>%
      mutate(
        independence = get(ro_indicator),
        .by = "iso_a3"
      ) %>%
      rename(country = iso_a3)

    if (ro_add_ecb_to_control) {
      data <- data %>%
        # Generate A ECB that is the average of its members when founded and unchanged since
        bind_rows(
          data %>%
            filter(year == 1998) %>%
            filter(country %in% c("AUT", "BEL", "FIN", "FRA", "DEU", "IRL", "ITA", "LUX", "NLD", "PRT", "ESP")) %>%
            summarise(
              across(-country, ~mean(.x))
            ) %>%
            complete(year = 1998:2023) %>%
            fill(-year, .direction = "down") %>%
            mutate(country = "EA")
        )
    }
  }

  if (dataset == "ga") {
    if ("CBI_2025_websiteGarriga.xlsx" %in% names(cache)) {
      data <- cache[["CBI_2025_websiteGarriga.xlsx"]]
    } else {
      data <- read_excel("./data/input/metadata/central_bank/CBI_2025_websiteGarriga.xlsx", sheet = "Sheet1")
      cache[["CBI_2025_websiteGarriga.xlsx"]] <<- data
    }
    data <- data %>%
      mutate(
        ccodewb = case_match(ccodewb, 891 ~ 688, 890 ~ 688, .default = ccodewb),
        country = countrycode(ccodewb, "iso3n", "iso3c")
      ) %>%
      select(country, year, all_of((c("lvau_garriga", "lvaw_garriga")))) %>%
      mutate(
        independence = get(ga_indicator),
        .by = "country"
      ) %>%
      filter(!is.na(country))

    if (ga_add_ecb_to_control) {
      data <- data %>%
        # Generate A ECB that is the average of its members when founded and unchanged since
        bind_rows(
          data %>%
            filter(year == 1998) %>%
            filter(country %in% c("AUT", "BEL", "FIN", "FRA", "DEU", "IRL", "ITA", "LUX", "NLD", "PRT", "ESP")) %>%
            summarise(
              across(-country, ~mean(.x))
            ) %>%
            complete(year = 1998:2012) %>%
            fill(-year, .direction = "down") %>%
            mutate(country = "EA")
        )
    }
  }

  if (dataset == "maro_supervision") {
    data <- read_stata("./data/input/metadata/central_bank/CBISIndex.dta") %>%
    rename("cbis" = "CBISIndex", "cbis_res" = "CBISBisIndex", "country" = "iso_a3") %>%
    select(country, year, cbis, cbis_res) %>%
      mutate(
        independence = cbis
      )
  }

  if (!is.null(limit_countries)) {
    data <- data %>% filter(
      country %in% limit_countries
    )
  }

  if (!is.null(forward_fill_until)) {
    missing_years <- (max(data$year) + 1):(forward_fill_until)
    combinations <- expand.grid(country = unique(data$country), year = missing_years)
    data <- data %>%
      bind_rows(
        combinations
      ) %>%
      arrange(country, year) %>%
      group_by(country) %>%
      fill(independence, .direction = "down") %>%
      ungroup()
  }

  data %>%
    select(country, year, independence) %>%
    arrange(country, year) %>%
    mutate(
      diff = independence - lag(independence),
      pct = (independence - lag(independence)) / lag(independence),
      .by = "country"
    ) %>%
    mutate(
      year = year + offset_treatmnet,
    ) %>%
    filter(year %in% range_years) %>%
    mutate(
      change = diff != 0,
      increase = diff > 0,
      decrease = diff < 0,
      .by = "country"
    ) %>%
    mutate(
      intensity = case_when(
        flip == TRUE ~ -1 * diff,
        TRUE ~ diff
      ),
      intensity = ifelse(abs(intensity) < minimum_intensity, 0, intensity),
      .by = "country"
    ) %>%
  {
    if (negative_strategy == "zero") {
      mutate(
        ., intensity = ifelse(intensity < 0, 0, intensity),
      )
    } else if (negative_strategy == "drop") {
      filter(., !any(intensity < 0), .by = "country")
    } else . # allow case is covered by this
  } %>%
  {
    if (!is.null(top_x)) {
      top_x_treatments <- sort(.$intensity) %>% tail(top_x)
      lowest_top_treatment <- top_x_treatments[1]
      mutate(.,
             intensity = ifelse(intensity < lowest_top_treatment, 0, intensity)
      )
    } else .
  } %>%
  {
    if (allow_multiple_treatments == FALSE && which == "drop") {
      mutate(.,
             multiplpe_treatments = sum(intensity != 0, na.rm = TRUE) > 1,
             .by = "country"
      ) %>%
        filter(multiplpe_treatments == FALSE)
    } else .
  } %>%
    filter(!is.na(intensity)) %>% # Drop missinig intensity. These are first period values. If balancing is enabled it will be readded.
  {
    if (balance_panel_strategy == "missing") {
      complete(., country, year)
    } else if (balance_panel_strategy == "zero") {
      complete(., country, year) %>%
        mutate(
          intensity = ifelse(is.na(intensity), 0, intensity)
        )
    } else . # Covers base unbalanced case
  } %>%
    mutate(
      number_of_increases = cumsum(intensity > 0),
      number_of_changes = cumsum(intensity != 0),
      number_of_decreases = cumsum(intensity < 0),
      treatment_year = case_when(
        which == "max" ~ find_treatment_year(year, intensity, how = "max", resolve_equal_intensity = "earlier"),
        which == "first" ~ find_treatment_year(year, intensity, how = "first"),
        which == "last" ~ find_treatment_year(year, intensity, how = "last"),
        which == "drop" ~ find_treatment_year(year, intensity, how = "first"), # Doesnt matter how, it is unique
        which == "closest" ~ find_closest_year(year, intensity),
        TRUE ~ NA
      ),
      treating = case_when(
        allow_multiple_treatments == FALSE & is.na(never_treat_treatment_year) ~ !is.na(year == treatment_year),
        allow_multiple_treatments == FALSE & !is.na(never_treat_treatment_year) ~ year == treatment_year,
        TRUE ~ intensity != 0
      ),
      ever_treated = isTRUE(any(treating == 1)), # Wrap with isTRUE to assign na and false the value false
      intensity = case_when(
        allow_multiple_treatments == FALSE ~ intensity * treating, # Set all intensities to zero that are not in the treatment year
        TRUE ~ intensity
      ),
      intensity = case_when(
        is.null(cumulate_intensity) == TRUE ~ intensity,
        !is.null(cumulate_intensity) && cumulate_intensity == "zero" ~ cumsum(coalesce(intensity, 0)),
        !is.null(cumulate_intensity) && cumulate_intensity == "absolute" ~ fill_upwards(independence)
      ),
      time_post_treat = case_when(
        ever_treated ~ year - treatment_year,
        TRUE ~ never_treat_time_post_treat # This is the value that never treated observations get in the time_post_treatment variable
      ),
      treated = case_when(
        ever_treated ~ time_post_treat >= 0, # Treatment year can be zero,
        TRUE ~ FALSE
      ), # When allowing for multiple treatments these are still relative to what is specified in which
      .by = "country"
    ) %>%
    mutate(across(c(treated, treating, ever_treated), ~as.numeric(.x))) %>%
    select(country, year, treatment_year, treating, intensity, treated, time_post_treat, ever_treated, number_of_increases)
}

#' Randomize Treatments in a Dataset
#'
#' This function takes a dataframe produced by generate_treatment and randomizes
#' the occurred treatments, either by randomizing dates or both treated countries and dates.
#'
#' @param data Data frame. Input data produced by generate_treatment.
#' @param how Character. Options: "dates" (randomizes dates) or "both" (randomizes treated countries and dates).
#' @param randomize_seed Numeric. Optional seed for randomization.
#' @param allow_multiple_treatments_per_country Logical. Whether to allow multiple treatments per country.
#' @param never_treat_treatment_year Numeric. treatment_year value for never-treated units.
#' @param never_treat_time_post_treat Numeric. time_post_treat value for never-treated units.
#'
#' @return A data frame with randomized treatment variables.
randomize_treatments <- function(
  data,
  how = "both",
  randomize_seed = 123,
  allow_multiple_treatments_per_country = FALSE,
  never_treat_treatment_year = 0,
  never_treat_time_post_treat = -9999
) {
  set.seed(randomize_seed)
  if (how == "dates") {
    # Treated countries remaing the same only change date:
    never_treated <- data %>% filter(ever_treated == 0)
    ever_treated <- data %>% filter(ever_treated == 1)
    ever_treated <- ever_treated %>%
      slice_sample(prop = 1, by = country) %>%
      mutate(
        year = sort(year), .by = "country"
      )

    randomized <- bind_rows(
      never_treated,
      ever_treated
    )

  }
  if (how == "both") {
    # Randomize the countries that are treated:
    countries <- data$country %>% unique()

    # Number of treatments
    treatments <- sum(data$treating == 1, na.rm = T)

    # Sample countries that were treated:
    treated_countries <- sample(countries, size = treatments)

    randomized <-
      data %>%
        mutate(
          treating = case_when(
            country[1] %in% treated_countries ~ sample(c(1, rep(0, length(treating) - 1))),
            TRUE ~ 0
          ),
          ever_treated = case_when(
            country[1] %in% treated_countries ~ 1,
            TRUE ~ 0
          ),
          .by = "country"
        )

    # Assign a random intensity from the available intensities
    randomized[randomized$treating == 1, "intensity"] <- sample(data %>%
                                                                  filter(treating == 1) %>%
                                                                  pull(intensity))
  }

  randomized %>% mutate(
    treatment_year = if_else(
      any(treating == 1),
      year[which.max(treating == 1)], # Need to use which max here because the ifelse appratnly evaluates this part even if the any part is false
      never_treat_treatment_year
    ),
    time_post_treat = case_when(
      any(treating == 1) ~ year - treatment_year,
      TRUE ~ never_treat_time_post_treat
    ),
    treated = case_when(
      any(treating == 1) ~ year >= treatment_year,
      TRUE ~ 0
    ),
    .by = "country"
  )
}

#' Calculate Lagged Values with Binning
#'
#' This function calculates lagged values of a series with optional binning for the last lag.
#'
#' @param series Numeric vector. The input series.
#' @param lag Numeric. The number of lags.
#' @param fill Any. Value to fill for non-observable lags.
#'
#' @return A numeric vector of lagged values.
bin_lag <- function(series, lag = 5, fill = NA) {

  # Fixes bug if a larger lag is requested than observable
  if (lag >= length(series)) {
    return(rep(fill, length(series)))
  }

  lagged_vector <- dplyr::lag(series, lag)
  observable_part <- lagged_vector[(lag + 1):length(series)]

  if (!is.na(series[1])) {
    return(c(rep(fill, lag), cumsum(replace_na(observable_part, 0))))
  } else {
    # If the vector starts with NA we set all to na up the first non-na value
    first_non_na <- min(which(!is.na(series)))
    return_value <- c(rep(fill, lag), cumsum(replace_na(observable_part, 0)))
    return_value[1:(lag + first_non_na - 1)] <- NA
    return(return_value)
  }
}

#' Calculate Leading Values with Binning
#'
#' This function calculates leading values of a series with optional binning for the first lead.
#'
#' @param series Numeric vector. The input series.
#' @param lead Numeric. The number of leads.
#' @param fill Any. Value to fill for non-observable leads.
#'
#' @return A numeric vector of leading values.
bin_lead <- function(series, lead = 5, fill = NA) {

  # Fixes bug if a larger lag is requested than observable
  if (lead >= length(series)) {
    return(rep(fill, length(series)))
  }

  leaded_vector <- dplyr::lead(series, lead)
  observable_part <- leaded_vector[1:(length(series) - lead)]

  if (!is.na(series[length(series)])) {
    return(c(rev(cumsum(rev(replace_na(observable_part, 0)))), rep(fill, lead)))
  } else {
    last_non_na <- max(which(!is.na(series)))
    return_value <- c(rev(cumsum(rev(replace_na(observable_part, 0)))), rep(fill, lead))
    return_value[(last_non_na + 1 - lead):(length(series))] <- NA
    return(return_value)
  }
}

#' Calculate Lagged Values with NA Awareness
#'
#' This function calculates lagged values of a series with special handling filling with NAs. If the series starts with NA's
#' it does fill with NA and not with what is provided under fill. Useful for filling unobserved lags but not observations that are explicitely missing.
#'
#' @param series Numeric vector. The input series.
#' @param lag Numeric. The number of lags.
#' @param fill Any. Value to fill for non-observable lags.
#'
#' @return A numeric vector of lagged values.
lag_na_aware <- function(series, lag = 1, fill = NA) {
  # The series starts with missing values, we fill with NA always as otherwise this would introduce 0 where there are no observations
  if (is.na(series[1])) {
    return(dplyr::lag(series, lag, NA))
  } else if (!is.na(fill) && fill == "last") {
    lagged_vector <- dplyr::lag(series, lag, NA)
    lagged_vector[1:lag] <- lagged_vector[lag + 1]
    return(lagged_vector)
  } else {
    return(dplyr::lag(series, lag, fill))
  }
}

#' Calculate Leading Values with NA Awareness
#'
#' This function calculates leading values of a series with special handling for NA values. If the series ends with NA's
#' it does fill with NA and not with what is provided under fill. Useful for filling unobserved leads but not observations that are explicitely missing
#'
#' @param series Numeric vector. The input series.
#' @param lead Numeric. The number of leads.
#' @param fill Any. Value to fill for non-observable leads.
#'
#' @return A numeric vector of leading values.
lead_na_aware <- function(series, lead = 1, fill = NA) {
  if (is.na(series[length(series)])) {
    return(dplyr::lead(series, lead, NA))
  } else if (!is.na(fill) && fill == "last") {
    leaded_vector <- dplyr::lead(series, lead, NA)
    leaded_vector[(length(leaded_vector) - lead + 1):length(leaded_vector)] <- leaded_vector[(length(leaded_vector) - lead)]
    return(leaded_vector)
  }
  else {
    return(dplyr::lead(series, lead, fill))
  }
}

#' Calculate Lags or Leads for a Variable in Panel Data
#'
#' This function calculates lags or leads for a specified variable in panel data.
#'
#' @param data Data frame. The input panel data.
#' @param variable Character. The name of the variable to lag/lead.
#' @param n Numeric vector. The number of lags/leads to calculate.
#' @param type Character. Either "lag" or "lead".
#' @param fill Any. Value to fill for non-observable lags/leads.
#' @param bin_last Logical. Whether to bin the last lag/lead.
#' @param id_group Character. The name of the ID variable for the panel.
#' @param time_group Character. The name of the time variable for the panel.
#'
#' @return A data frame with calculated lags or leads added.
calculate_lags <- function(data, variable, n = 1, type = "lag", fill = 0, bin_last = TRUE, id_group = "country", time_group = "year") {

  # This removes duplicate observations that share a country, year id.
  panel_level <- data %>%
    arrange(!!!rlang::syms(c(id_group, time_group))) %>%
    group_by(across(all_of(c(id_group, time_group)))) %>%
    summarise(
      var = get(variable)[1],
    ) %>%
    ungroup()

  if (bin_last) {
    max_n <- tail(n, 1)
    n <- head(n, -1) # Remove last
    if (type == "lag") {
      last_lag <- panel_level %>%
        mutate("l{max_n}" := bin_lag(var, max_n, fill), .by = all_of(id_group), .keep = "none") %>%
        .[, 2] %>%
        rename_with(~paste0(.x, "_", variable))
    } else if (type == "lead") {
      last_lead <- panel_level %>%
        mutate("d{max_n}" := bin_lead(var, max_n, fill), .by = all_of(id_group), .keep = "none") %>%
        .[, 2] %>%
        rename_with(~paste0(.x, "_", variable))
    }
  } else {
    last_lead <- NULL
    last_lag <- NULL
  }

  if (type == "lag") {
    lags <- bind_cols(
      panel_level %>% select(-var),
      map_dfc(n, ~panel_level %>%
        mutate("l{.x}" := lag_na_aware(var, .x, fill = fill), .by = all_of(id_group), .keep = "none") %>%
        .[, 2]) %>% rename_with(~paste0(.x, "_", variable)),
      last_lag
    )
  } else if (type == "lead") {
    lags <- bind_cols(
      panel_level %>% select(-var),
      map_dfc(n, ~panel_level %>%
        mutate("d{.x}" := lead_na_aware(var, .x, fill = fill), .by = all_of(id_group), .keep = "none") %>%
        .[, 2]) %>% rename_with(~paste0(.x, "_", variable)),
      last_lead
    )
  }
  data %>% left_join(lags, by = c(id_group, time_group))
}

############## IMPLEMENT DIFF IN DIFF ESTIMATOR #########################

twfe_binary <- function(y, data, continueous_treatments = FALSE, controls = NULL, interact = NULL, leads = "all", lags = "all", bin_last = TRUE, fill = 0, unit_trends = FALSE, overwrite_model_name = NULL) {
  if (leads == "all" || lags == "all") {
    if (continueous_treatments || !is.null(interact)) {
      warning("leads = all or lags = all does not support interactions terms or multiple treatments")
    }
  }

  if (continueous_treatments == TRUE) {
    # Estimate using intensity rather than treatment indictor
    treatment_variable <- "intensity"
  } else {
    treatment_variable <- "treating"
  }

  if (is.null(controls)) {
    controls_formula <- ""
  } else {
    controls_formula <- paste("+", paste0(controls, collapse = " + "))
  }

  # Interacts with a variable.
  if (!is.null(interact)) {
    interacted_leads <- paste0("d", (leads + 1):2, paste0("_", treatment_variable, " : ", interact), collapse = " + ")
    interacted_lags <- paste0("l", 0:lags, paste0("_", treatment_variable, " : ", interact), collapse = " + ")
    interactions_formula <- paste("+", interacted_leads, " + ", interacted_lags)
  } else {
    interactions_formula <- ""
  }

  if (unit_trends) {
    unit_trends <- "+ country:time_post_treat"
  } else {
    unit_trends <- ""
  }

  if (leads == "all" | lags == "all") {
    formula_str <- str_glue("{y} ~ i(time_post_treat, ref = c(-1, -9999)) {controls_formula} {unit_trends} | country + year") # This is the simple case but cannot accommodate more than one treatment
    if ((leads == "all" & lags != "all") | ((leads != "all" & lags == "all"))) {
      warning("When using the all setting both lags and leads should to be set to all. Using all leads and lags now")
    }
  } else {
    # First manually calculate leads.
    data <- data %>%
      calculate_lags(treatment_variable, n = 2:(leads + 1), type = "lead", bin_last = bin_last, fill = fill) %>%
      calculate_lags(treatment_variable, n = 0:lags, type = "lag", bin_last = bin_last, fill = fill)

    leads_formula <- paste0("d", (leads + 1):2, "_", treatment_variable, collapse = " + ")
    lags_formula <- paste0("l", 0:lags, "_", treatment_variable, collapse = " + ")

    formula_str <- str_glue("{y} ~ {leads_formula} + {lags_formula} {controls_formula} {interactions_formula} {unit_trends} | country + year") # This treats the year as one before treatment as the base

  }

  twfe <- feols(as.formula(formula_str), data = data, cluster = ~country)


  if (leads == "all" | lags == "all") {
    estimates <- tibble(beta = twfe$coefficients, se = twfe$se) %>% mutate(
      event_time = as.numeric(str_extract(names(twfe$coefficients), "-?\\d+")),
      type = case_when(
        str_detect(names(twfe$coefficients), "^time_post_treat::") ~ "es_coef",
        TRUE ~ "control"
      ),
      .before = beta
    )
  } else {
    estimates <- tibble(beta = twfe$coefficients, se = twfe$se) %>%
      mutate(
        event_time = str_replace_all(names(twfe$coefficients), c("^l" = "", "^d" = "-")) %>%
          str_extract("^-?\\d+") %>%
          as.numeric(),
        type = case_when(
          str_detect(names(twfe$coefficients), "^(l|d)\\d+.+:") ~ "interaction",
          str_detect(names(twfe$coefficients), "^(l|d)\\d+") ~ "es_coef",
          TRUE ~ "control"
        ),
        .before = beta) %>%
      arrange(event_time)
  }

  return(
    list(
      "estimates" = estimates,
      "estimated_model" = twfe,
      "model_name" = ifelse(is.null(overwrite_model_name), "Two-way fixed effects", overwrite_model_name),
      "y" = y,
      "vcov" = vcov(twfe),
      "coef" = coef(twfe),
      "nobs" = nobs(twfe)
    ))
}

#' Sun & Abraham (2021) Event Study Estimation
#'
#' This function implements the Sun & Abraham (2021) event study estimation method.
#'
#' @param y Character. The name of the outcome variable.
#' @param data Data frame. The input panel data.
#' @param controls Character vector or NULL. Names of control variables.
#' @param leads Character or numeric. Number of lead periods or "all" for all available.
#' @param lags Character or numeric. Number of lag periods or "all" for all available.
#' @param bin_offset Numeric. Offset for binning periods outside of the observation window periods. 1 to bin inside. 2 to bin outside
#'
#' @return A list containing estimates, estimated model, model name, and outcome variable name.
sun_abraham <- function(y, data, controls = NULL, leads = "all", lags = "all", bin_offset = 2) {

  if (leads != "all" || lags != "all") {
    bin <- str_glue(", bin.rel = c({-leads - bin_offset}:-100, {lags + bin_offset}:100)") # 100 is a arbitrary number larger than the max/min time to treat to ensure everyting outside of the
    # leads and lags is binned into one variable
  } else if (leads != "all") {
    bin <- str_glue(", bin.rel = {-leads -bin_offset}:-100")
  }  else if (leads != "all") {
    bin <- str_glue(", bin.rel = {lags + bin_offset}:100")
  } else {
    # No binning
    bin <- ""
  }

  if (is.null(controls)) {
    controls_formula <- ""
  } else {
    controls_formula <- str_glue(" + {cont}", cont = paste0(controls, collapse = " + "))
  }

  subab_formula <- str_glue('{y} ~ sunab(treatment_year, time_post_treat,  ref.c = 0, {bin}) {controls_formula} | country + year')

  subab_est <- feols(as.formula(subab_formula), data = data, cluster = ~country)

  # Broom seems to correctly produce the time to treat eastimates:
  estimates <- broom::tidy(subab_est) %>%
    mutate(
      beta = estimate,
      se = std.error,
      event_time = as.numeric(str_extract(term, "-?\\d+")),
      type = "es_coef"
    ) %>%
    select(event_time, type, beta, se) %>%
  {
    if (leads != "all" && leads != "all") {
      filter(., -leads - 2 < event_time, event_time < lags + 1)
    }
    else if (leads != "all") {
      filter(., -leads - 2 < event_time)
    }
    else if (lags != "all") {
      filter(., event_time < lags + 1)
    } else .
  }

  return(
    list(
      "estimates" = estimates,
      "estimated_model" = subab_est,
      "model_name" = "Sun & Abraham (2021)",
      "y" = y
    ))
}


#' Gardner et al. (2024) Two-Stage Difference-in-Differences Estimation
#'
#' This function implements the Gardner et al. (2024) two-stage difference-in-differences estimation method.
#'
#' @param y Character. The name of the outcome variable.
#' @param data Data frame. The input panel data.
#' @param controls Character vector or NULL. Names of control variables.
#' @param leads Character or numeric. Number of lead periods or "all" for all available.
#' @param lags Character or numeric. Number of lag periods or "all" for all available.
#' @param bin_last Logical. Whether to bin the last lag/lead.
#' @param fill Any. Value to fill for non-observable lags/leads.
#' @param overwrite_model_name Character or NULL. Custom name for the model.
#' @param bootstrap Numeric or NULL. Number of bootstrap iterations.
#'
#' @return A list containing estimates, estimated model, model name, outcome variable name, coefficients, variance-covariance matrix, and number of observations.
gardner_did <- function(y, data, controls = NULL, leads = "all", lags = "all", bin_last = TRUE, fill = 0, overwrite_model_name = NULL, bootstrap = NULL) {
  if (is.null(controls)) {
    first_stage_formula_str <- "~ 0 | country + year"
  } else {
    controls_formula <- paste0(controls, collapse = " + ")
    first_stage_formula_str <- str_glue("~ {controls_formula} | country + year")
  }

  if (leads == "all" | lags == "all") {
    main_formula_str <- str_glue("~ i(time_post_treat, ref = c(-1, -9999))") # This is the simple case but cannot accommodate more than one treatment
    if ((leads == "all" & lags != "all") | ((leads != "all" & lags == "all"))) {
      warning("When using the all setting both lags and leads should to be set to all. Using all leads and lags now")
    }
  } else {
    # First manually calculate leads.
    data <- data %>%
      calculate_lags("treating", n = 2:(leads + 1), type = "lead", bin_last = bin_last, fill = fill) %>%
      calculate_lags("treating", n = 0:lags, type = "lag", bin_last = bin_last, fill = fill)

    # Manually drop observations that contain missing values. Otherwise the estimation fails
    if (is.na(fill)) {
      data <- data %>% drop_na(matches("^(d|l)\\d+_treating")) # Drop all that are not completely observed
    }

    leads_formula <- paste0("d", (leads + 1):2, "_treating", collapse = " + ")
    lags_formula <- paste0("l", 0:lags, "_treating", collapse = " + ")

    main_formula_str <- str_glue("~ {leads_formula} + {lags_formula}") # This treats the year as one before treatment as the base
  }

  did2s <- did2s(data,
                 yname = y,
                 first_stage = as.formula(first_stage_formula_str),
                 second_stage = as.formula(main_formula_str),
                 bootstrap = ifelse(is.null(bootstrap), FALSE, TRUE),
                 n_bootstraps = bootstrap,
                 treatment = "treated",
                 cluster_var = "country")


  if (leads == "all" | lags == "all") {
    estimates <- tibble(beta = did2s$coefficients, se = did2s$se) %>% mutate(
      event_time = as.numeric(str_extract(names(did2s$coefficients), "-?\\d+")),
      type = "es_coef",
      .before = beta
    )
  } else {
    estimates <- tibble(beta = did2s$coefficients, se = did2s$se) %>%
      mutate(
        event_time = str_replace_all(names(did2s$coefficients), c("^l" = "", "^d" = "-")) %>%
          str_extract("^-?\\d+") %>%
          as.numeric(),
        type = "es_coef", # Coefficients for the control variables are not present here
        .before = beta) %>%
      arrange(event_time)
  }

  return(
    list(
      "estimates" = estimates,
      "estimated_model" = did2s,
      "model_name" = ifelse(is.null(overwrite_model_name), "Gardner et al. (2024)", overwrite_model_name),
      "y" = y,
      "coef" = coef(did2s),
      "vcov" = vcov(did2s),
      "nobs" = nobs(did2s)
    ))
}

#' Freyaldenhoven et al. TWFE Event Study Estimation
#'
#' This function implements a two-way fixed effects event study estimation based on Freyaldenhoven et al.'s eventstudyr package.
#' Should be (and is) identical to twfe_binary
#'
#' @param y Character. The name of the outcome variable.
#' @param data Data frame. The input panel data.
#' @param leads Numeric. Number of lead periods.
#' @param lags Numeric. Number of lag periods.
#' @param t_band_conf Numeric. Confidence level for t-bands.
#' @param t_band_seed Numeric. Seed for t-band calculation.
#' @param overwrite_model_name Character or NULL. Custom name for the model.
#'
#' @return A list containing estimates, estimated model, model name, outcome variable name, coefficients, variance-covariance matrix, and number of observations.
twfe_freyaldenhoven <- function(y, data, leads, lags, t_band_conf = 90, t_band_seed = 1, overwrite_model_name = NULL) {
  # Note this package doesn't work if the treatment varies on a higher level than the observations. So it only works for country-year aggregations
  # Further this model also returns t bands and on top of standard errors. It does standard "twfe" estimation with endpoint binning.
  # Coefficients are the same as equivalent specifications estimated with twfe_binary or twfe_distributed lag

  # Make copy of data because the package declares the dataframe to DT (in place) which messes up the tibble printing and
  # might have other unintended consequences elsewhere
  data_copy <- data

  est_event_study <- eventstudyr::EventStudy(
    estimator = "OLS",
    data = data_copy,
    outcomevar = y,
    policyvar = "treated", # Here we need to use the treated variable
    idvar = "country",
    timevar = "year",
    post = lags - 1, # Need to subtract one to be consistent with the other functions in this package lags count all coefficients and the last is (the binned one) is counted seperately
    pre = 0,
    overidpre = leads
  )

  set.seed(1)
  # Hope that the package internals do not change as this function is not exported by the package
  estimates <- eventstudyr:::AddSuptBand(est_event_study$output, num_sim = 1000, conf_level = 0.95, eventstudy_coefficients = est_event_study$arguments$eventstudy_coefficients) %>% tibble()

  rename_estimates <- function(est_names,
                               return_what = "event_time" # Either directly returns event time or the assosciated coefficient names
  ) {

    # Some ugly code to get the event times from the coefficient names
    est_names <- str_replace_all(est_names, c("lead" = "-", "lag" = ""))
    extract_event_time <- str_extract(est_names, "(?<=_)(-|\\d)+$")
    event_time <- case_when(
      str_detect(est_names, "_fd_") ~ as.numeric(extract_event_time),
      str_detect(extract_event_time, "-") ~ as.numeric(extract_event_time) - 1, # Last need to subtract one to match binning definition,
      is.na(extract_event_time) ~ 0,
      TRUE ~ as.numeric(extract_event_time)
    )

    coef_names_convention <- case_when(
      event_time < 0 ~ str_glue("d{abs(event_time)}_treating"),
      event_time >= 0 ~ str_glue("l{event_time}_treating")
    )

    if (return_what == "event_time") {
      return(event_time)
    } else {
      return(coef_names_convention)
    }
  }

  estimates <- estimates %>%
    mutate(
      beta = estimate,
      se = std.error,
      event_time = rename_estimates(term, return_what = "event_time"),
      type = "es_coef"
    ) %>%
    select(
      event_time, type, beta, se, suptband_lower, suptband_upper
    )

  # Correctly named coefficient and vcov matrix
  coef_names <- rename_estimates(names(est_event_study$output$coefficients), return_what = "coef_names")
  coefs <- est_event_study$output$coefficients %>% set_names(coef_names)

  var_cov <- est_event_study$output$vcov
  rownames(var_cov) <- coef_names
  colnames(var_cov) <- coef_names


  return(
    list(
      "estimates" = estimates,
      "estimated_model" = est_event_study$output,
      "model_name" = ifelse(is.null(overwrite_model_name), "TWFE Freyaldenhoven", overwrite_model_name),
      "y" = y,
      "coef" = coefs,
      "vcov" = var_cov,
      "nobs" = est_event_study$output$nobs
    ))
}

#' Stacked Difference-in-Differences Estimation
#'
#' This function implements a stacked difference-in-differences estimation method.
#'
#' @param y Character. The name of the outcome variable.
#' @param data Data frame. The input panel data.
#' @param leads Numeric. Number of lead periods.
#' @param lags Numeric. Number of lag periods.
#' @param fill Any. Value to fill for non-observable lags/leads.
#' @param comparsion Character. Strategy for comparison group: "never_treated", "not_yet_treated", or "not_yet_and_nevertreated".
#' @param overwrite_model_name Character or NULL. Custom name for the model.
#'
#' @return A list containing estimates, estimated model, model name, outcome variable name, variance-covariance matrix, coefficients, and number of observations.
stacked_did <- function(y, data, leads = 7, lags = 14, fill = 0, comparsion = "never_treated", overwrite_model_name = NULL) {
  years_with_treatments <- data %>%
    filter(treating == 1) %>%
    pull(year) %>%
    unique()

  stack_exp <- years_with_treatments %>%
    map_dfr(function(x) {
      data %>%
      {
        if (comparsion == "never_treated") {
          filter(., treatment_year == x | treatment_year == 0) # Only compare against never treated
        } else if (comparsion == "not_yet_treated") {
          filter(., treatment_year >= x) #  Include later treated (needs to be checked for correctness). Problem could be that these are also treated eventually and we have the early vs. later problem again.
        } else if (comparsion == "not_yet_and_nevertreated") {
          filter(., treatment_year >= x | treatment_year == 0) # Same remark as above
        }
      } %>%
        mutate(
          cohort = x
        )
    }) %>%
    mutate(
      stack_year = paste0(cohort, "_", year),
      stack_country = paste0(country, "_", country)
    ) %>%
    calculate_lags("treating", n = 2:(leads + 1), type = "lead", bin_last = T, fill = fill) %>%
    calculate_lags("treating", n = 0:lags, type = "lag", bin_last = T, fill = fill)

  leads_formula <- paste0("d", (leads + 1):2, "_", "treating", collapse = " + ")
  lags_formula <- paste0("l", 0:lags, "_", "treating", collapse = " + ")

  if (leads == "all" | lags == "all") {
    formula_str <- str_glue("{y} ~ i(time_post_treat, ref = c(-1, -9999)) | stack_year + stack_country") # This is the simple case but cannot accommodate more than one treatment
  } else {
    formula_str <- str_glue("{y} ~ {leads_formula} + {lags_formula} | stack_year + stack_country")
  }

  stack_did <- feols(fml = as.formula(formula_str), data = stack_exp, cluster = ~stack_country)


  if (leads == "all" | lags == "all") {
    estimates <- tibble(beta = stack_did$coefficients, se = stack_did$se) %>% mutate(
      event_time = as.numeric(str_extract(names(stack_did$coefficients), "-?\\d+")),
      type = case_when(
        str_detect(names(stack_did$coefficients), "^time_post_treat::") ~ "es_coef",
        TRUE ~ "control"
      ),
      .before = beta
    )
  } else {
    estimates <- tibble(beta = stack_did$coefficients, se = stack_did$se) %>%
      mutate(
        event_time = str_replace_all(names(stack_did$coefficients), c("^l" = "", "^d" = "-")) %>%
          str_extract("^-?\\d+") %>%
          as.numeric(),
        type = case_when(
          str_detect(names(stack_did$coefficients), "^(l|d)\\d+") ~ "es_coef",
          TRUE ~ "control"
        ),
        .before = beta) %>%
      arrange(event_time)
  }

  return(
    list(
      "estimates" = estimates,
      "estimated_model" = stack_did,
      "model_name" = ifelse(is.null(overwrite_model_name), "Stacked DiD", overwrite_model_name),
      "y" = y,
      "vcov" = vcov(stack_did),
      "coef" = coef(stack_did),
      "nobs" = nobs(stack_did)
    ))
}


#' Two-Way Fixed Effects Distributed Lag Model
#'
#' This function implements a two-way fixed effects estimation using distributed lags. Should be identical to twfe_binary if endpoints are correctly binned
#' Requires intensity to be passed (as generated by generate_treatment)
#'
#' @param y Character. The name of the outcome variable.
#' @param data Data frame. The input panel data.
#' @param leads Numeric. Number of lead periods.
#' @param lags Numeric. Number of lag periods.
#' @param fill Any. Value to fill for non-observable lags/leads.
#'
#' @return A list containing estimates, estimated model, model name, and outcome variable name.
twfe_distributed_lag <- function(y, data, leads = 7, lags = 14, fill = 0) {
  # Fill decides what to do if a value is not observed. To be consistent here with the event study estimates
  # fill = 0, actually fills with the last observed value, which is consistent with the notion of filling
  # with 0s in the event studies which assume no treatment

  # Define local functions for aggregating distributed lag models into event study coefficients taken from the replication code
  # of Schmidheiny and Siegloch (2023)

  # function for reverse cumulative sum of vector
  revcumsum <- function(x) {
    x <- rev(cumsum(rev(x)))
  }

  # function to calculate standard errors of cumulative sum
  secumsum <- function(vcov) {
    L <- dim(vcov)[1]
    se <- c()
    for (i in c(1:L)) {
      a <- matrix(rep(1, i), nrow = 1)
      V <- a %*% vcov[1:i, 1:i] %*% t(a)
      se[i] <- sqrt(V)
    }
    return(se)
  }

  # function to calculate standard errors of reverse cumulative sum
  serevcumsum <- function(vcov) {
    L <- dim(vcov)[1]
    se <- c()
    for (i in c(L:1)) {
      a <- matrix(rep(1, L - i + 1), nrow = 1)
      V <- a %*% vcov[i:L, i:L] %*% t(a)
      se[i] <- sqrt(V)
    }
    return(se)
  }

  if (fill == 0) {
    fill <- "last"
  }

  data <- data %>%
    calculate_lags("intensity", n = 1:leads, type = "lead", bin_last = F, fill = fill) %>%
    calculate_lags("intensity", n = 0:lags, type = "lag", bin_last = F, fill = fill)

  leads_formula <- paste0("d", 1:leads, "_", "intensity", collapse = " + ")
  lags_formula <- paste0("l", 0:lags, "_", "intensity", collapse = " + ")

  formula_str <- str_glue("{y} ~ {leads_formula} + {lags_formula} | country + year")

  distributed_lag_twfe <- feols(as.formula(formula_str), data = data, cluster = ~country)

  estimates <- tibble(
    event_time = -(leads + 1):lags,
    beta = c(-revcumsum(coef(distributed_lag_twfe)[leads:1]), 0, cumsum(coef(distributed_lag_twfe)[(leads + 1):(leads + 1 + lags)])),
    se = c(-serevcumsum(vcov(distributed_lag_twfe)[leads:1, leads:1]), 0, secumsum(vcov(distributed_lag_twfe)[(leads + 1):(leads + 1 + lags), (leads + 1):(leads + 1 + lags)])),
    type = "es_coef"
  )

  return(list(
    "estimates" = estimates,
    "estimated_model" = distributed_lag_twfe,
    "model_name" = "Distributed Lag",
    "y" = y
  ))
}

#' Callaway & Sant'Anna (2021) Difference-in-Differences Estimation
#'
#' This function implements the Callaway & Sant'Anna (2021) difference-in-differences estimation method.
#'
#' @param y Character. The name of the outcome variable.
#' @param data Data frame. The input panel data.
#' @param leads Character or numeric. Number of lead periods or "all" for all available.
#' @param lags Character or numeric. Number of lag periods or "all" for all available.
#' @param overwrite_model_name Character or NULL. Custom name for the model.
#'
#' @return A list containing estimates, estimated model, model name, and outcome variable name.
cs_did <- function(y, data, leads = "all", lags = "all", overwrite_model_name = NULL) {
  if (leads == "all") {
    leads <- -Inf
  }
  if (lags == "all") {
    lags <- Inf
  }

  att <- att_gt(
    yname = y,
    tname = "year",
    gname = "treatment_year",
    data = data %>% mutate(
      treatment_year = case_when(
        treatment_year == -9999 ~ 0,
        TRUE ~ treatment_year
      )
    ),
    panel = FALSE,
    bstrap = TRUE,
    base_period = "varying",
    allow_unbalanced_panel = TRUE,
    clustervars = "country",
    control_group = "nevertreated"
  )

  aggte <- aggte(att,
                 type = "dynamic",
                 min_e = -leads,
                 max_e = lags,
                 na.rm = T)

  estimates <- tibble(event_time = aggte$egt, beta = aggte$att.egt, se = aggte$se.egt, type = "es_coef")

  return(list(
    "estimates" = estimates,
    "estimated_model" = att,
    "model_name" = ifelse(is.null(overwrite_model_name), "Callaway & Sant'Anna (2021)", overwrite_model_name),
    "y" = y
  ))
}

#' Difference-in-Differences with Imputation
#'
#' This function implements a difference-in-differences estimation method with imputation.
#'
#' @param y Character. The name of the outcome variable.
#' @param data Data frame. The input panel data.
#' @param leads Character or numeric. Number of lead periods or "all" for all available.
#' @param lags Character or numeric. Number of lag periods or "all" for all available.
#' @param overwrite_model_name Character or NULL. Custom name for the model.
#' @param fill Any. Value to fill for non-observable lags/leads.
#'
#' @return A list containing estimates, estimated model, model name, and outcome variable name.
didimputation <- function(y, data, leads = "all", lags = "all", overwrite_model_name = NULL, fill = 0) {
  horizon <- if (leads == "all") TRUE else seq(0, lags, 1)
  pretrends <- if (leads == "all") TRUE else seq(-(leads + 1), -2, 1)

  imput_est <- did_imputation(
    data = data %>% mutate(
      treatment_year = case_when(
        treatment_year == -9999 ~ 0,
        TRUE ~ treatment_year
      )
    ),
    yname = y,
    gname = "treatment_year",
    tname = "year",
    idname = "country",
    horizon = horizon,
    pretrends = pretrends
  )

  estimates <- imput_est %>%
    mutate(
      event_time = as.numeric(term),
      beta = estimate,
      se = std.error,
      type = "es_coef"
    ) %>%
    select(event_time, beta, se, type)

  return(list(
    "estimates" = estimates,
    "estimated_model" = imput_est,
    "model_name" = ifelse(is.null(overwrite_model_name), "Borusyak, Jaravel, and Spiess (2024)", overwrite_model_name),
    "y" = y
  ))
}

#' Plot Multiple Event Study Estimates
#'
#' This function creates a plot comparing multiple event study estimates in a single graph. Not multiplpe panel
#' for that use plot_dominance_panel.
#'
#' @param event_studies List. Contains event study results to be plotted.
#' @param ylim Numeric vector or NULL. Y-axis limits.
#' @param ybreaks Numeric vector or NULL. Custom y-axis breaks.
#' @param yround_digits Numeric. Number of digits to round y-axis labels.
#' @param dodge_amount Numeric. Amount to dodge overlapping points.
#' @param restrict_event_window Numeric vector or NULL. Limits the event window to plot.
#' @param add_tests Logical. Whether to add statistical test results to the plot.
#' @param tests_x,tests_y Numeric. X and Y coordinates for test results annotation.
#' @param tests_text_line_width,tests_font_size Numeric. Formatting for test results text.
#' @param zero_line_value Numeric or NULL. Value to display at y=0 line.
#' @param percent_scale Logical. Whether to use percentage scale for y-axis.
#' @param binned_endpoints Logical. Whether to use binned endpoints for x-axis labels.
#'
#' @return A ggplot object representing the combined event study estimates.
plot_multiple_estimates <- function(event_studies,
                                    ylim = NULL,
                                    ybreaks = NULL,
                                    yround_digits = 2,
                                    dodge_amount = 0.35,
                                    restrict_event_window = NULL,
                                    add_tests = F,
                                    tests_x = 0,
                                    tests_y = 0,
                                    tests_text_line_width = 0.35,
                                    tests_font_size = 1.75,
                                    tests_add_border = FALSE,
                                    zero_line_value = NULL,
                                    percent_scale = TRUE,
                                    binned_endpoints = TRUE
) {
  estimates_data <- event_studies %>%
    map_dfr(~.x[["estimates"]] %>%
      mutate(name = .x[["model_name"]]) %>%
      add_missing_base_coef) %>%
    fill(name, .direction = "down") %>%
  {
    if (!is.null(restrict_event_window)) {
      filter(., event_time >= restrict_event_window[1], event_time <= restrict_event_window[2])
    } else .
  }
  plot <- estimates_data %>%
    ggplot(aes(x = event_time, y = beta, shape = name, color = name, fill = name)) +
    geom_hline(yintercept = 0, linewidth = 0.25) +
    geom_vline(xintercept = -1, linetype = "dashed", linewidth = 0.25) +
    geom_point(position = position_dodge(width = dodge_amount)) +
    geom_errorbar(aes(ymin = beta - 1.96 * se, ymax = beta + 1.96 * se), position = position_dodge(width = dodge_amount), width = 0.5) +
    paper_theme() +
    labs(
      x = "Time to independence event",
      y = "Coefficient and 95% CI",
      shape = "",
      color = "",
      fill = ""
    )

  if (add_tests) {
    coef_tests <- event_studies %>% map_dfr(function(x) {
      tibble(
        parallel_trends = test_parallel_trends(x[["coef"]], x[["vcov"]], x[["nobs"]]),
        level_off = test_level_off(x[["coef"]], x[["vcov"]], x[["nobs"]]),
        model = x[["model_name"]]
      ) })

    shorten_name <- function(model_name) {
      if (model_name == "Two-way fixed effects") {
        return("TWFE")
      } else {
        return(str_replace(model_name, " \\(\\d+\\)", ""))
      }
    }

    tests_annotation <- coef_tests %>%
      pmap_chr(function(...) {
        cols <- tibble(...)
        str_glue("{PARALLEL_TRENDS_TEST_STRING} = {p_parallel} -- {LEVEL_OFF_TEST_STRING} = {p_leveloff} ({m})",
                 p_parallel = format_number(cols$parallel_trends),
                 p_leveloff = format_number(cols$level_off),
                 m = shorten_name(cols$model))
      }) %>%
      paste(collapse = "\n")

    plot <- plot + annotate(
      geom = "label",
      label = tests_annotation,
      x = tests_x,
      y = tests_y,
      hjust = 0,
      label.r = unit(2, "pt"),
      size = tests_font_size,
      lineheight = tests_text_line_width,
      color = "black",
      family = "XCharter Math",
      label.size = ifelse(tests_add_border, 1, NA)
    )
  }

  label_function_y <- function(label_numbers) {
    label_numbers %>% map_chr(function(x) {
      if (is.na(x)) {
        return(NA)
      }
      else if (x == 0 & !is.null(zero_line_value)) {
        if (percent_scale) {
          return(str_glue("0 ({n}%)", n = format(round(zero_line_value * 100, 1), nsmall = 1)))
        } else {
          return(str_glue("0 ({n})", n = format(round(zero_line_value, 2), nsmall = 1)))
        }
      } else {
        if (percent_scale) {
          return(str_glue("{n}%", n = format(round(x * 100, yround_digits - 2), nsmall = 0)))
        } else {
          return(str_glue("{n}", n = format(round(x, yround_digits), nsmall = 0)))
        }
      }
    })
  }

  label_function_x <- function(label_numbers) {
    if (binned_endpoints) {
      sapply(label_numbers, function(x) {
        if (x == min(label_numbers)) {
          return(str_glue("\u2264{min(label_numbers)}"))
        } else if (x == max(label_numbers)) {
          return(str_glue("\u2265{max(label_numbers)}"))
        } else {
          return(as.character(x))
        }
      })
    } else {
      return(as.character(label_numbers))
    }
  }


  if (!is.null(ybreaks) | !is.null(ylim)) {
    plot <- plot + scale_y_continuous(breaks = ybreaks, limits = ylim, labels = label_function_y, oob = scales::oob_keep)
  } else {
    plot <- plot + scale_y_continuous(n.breaks = 4, labels = label_function_y, oob = scales::oob_keep)
  }

  plot <- plot +
    scale_x_continuous(breaks = seq(min(estimates_data$event_time, na.rm = T), max(estimates_data$event_time, na.rm = T), by = 1), labels = label_function_x) +
    discrete_scale('shape', 'shape_d', function(n) c(21, 23, 22)[seq_len(n)]) +
    coord_cartesian(clip = 'off')

  plot
}

#' Plot Dominance Panel
#'
#' This function creates a panel of plots for different dominance measures and estimates.
#'
#' @param list_of_estimates List. Contains estimates for different dominance measures.
#' @param remove_facet_titles Logical. Whether to remove facet titles.
#' @param custom_limits_y List. Custom y-axis limits for each plot.
#' @param custom_breaks_y List. Custom y-axis breaks for each plot.
#' @param custom_breaks_x Numeric vector or NULL. Custom x-axis breaks.
#' @param binned_endpoints Logical. Whether to use binned endpoints for x-axis labels.
#'
#' @return A combined plot object representing the dominance panel.
plot_dominance_panel <- function(list_of_estimates,
                                 remove_facet_titles = F,
                                 custom_limits_y = list(),
                                 custom_breaks_y = list(),
                                 custom_breaks_x = NULL,
                                 binned_endpoints = FALSE
) {

  combined_estimates <- list_of_estimates %>%
    map_dfr(function(estimate) {
      estimate$estimates %>%
        mutate(
          y = estimate$y,
          model_name = estimate$model_name
        ) %>%
        group_by(y, model_name) %>%
        add_missing_base_coef() %>%
        ungroup() %>%
        fill(y, model_name, .direction = "down")
    }
    ) %>%
    mutate(
      y = ordered(y, levels = dom, labels = dom_label)
    ) %>%
    mutate(
      se_upper = beta + 1.96 * se,
      se_lower = beta - 1.96 * se
    ) %>%
    arrange(y)

  #  Function to label axis
  label_function_x <- function(label_numbers) {
    if (binned_endpoints) {
      sapply(label_numbers, function(x) {
        if (x == min(label_numbers)) {
          return(str_glue("\u2264{min(label_numbers)}"))
        } else if (x == max(label_numbers)) {
          return(str_glue("\u2265{max(label_numbers)}"))
        } else {
          return(as.character(x))
        }
      })
    } else {
      return(as.character(label_numbers))
    }
  }

  charts <- combined_estimates %>%
    group_by(y) %>%
    group_map(
      function(data, group) {
        ggplot(data = data, aes(x = event_time, y = beta, color = model_name, shape = model_name)) +
          geom_hline(yintercept = 0, linewidth = 0.25) +
          geom_vline(xintercept = -1, linetype = "dashed", linewidth = 0.25) +
          geom_errorbar(aes(ymin = se_lower, ymax = se_upper), linewidth = 0.4, width = 0.6) +
          geom_point(size = 1) +
          facet_wrap(~model_name, nrow = 1, scales = "free") +
          labs(
            x = "Time to independence event",
            y = "Coefficient",
            color = "",
            shape = ""
          ) +
          scale_y_continuous(limits = if (length(custom_limits_y) > 0) custom_limits_y[[pull(group)]],
                             breaks = if (length(custom_breaks_y) > 0) custom_breaks_y[[pull(group)]],
                             labels = scales::percent
          ) +
          scale_x_continuous(
            breaks = if (!is.null(custom_breaks_x)) custom_breaks_x,
            labels = label_function_x
          ) +
          paper_theme()
      }
    )
  chart_labels <- unique(combined_estimates$y)

  charts <- charts %>% imap(
    ~.x + ggtitle(str_glue("{toupper(letters[.y])}. {chart_labels[.y]}"))
  )

  combined_chart <- wrap_plots(
    charts, ncol = 1, guides = "collect"
  ) & theme(
    legend.position = "bottom",
    axis.title.x = element_text(margin = margin(t = 4, r = 0, b = 0, l = 0)),
    plot.title = element_text(margin = margin(t = 0, r = 0, b = 7, l = 0))
  )

  if (remove_facet_titles) {
    combined_chart <- combined_chart &
      theme(
        strip.background = element_blank(),
        strip.text.x = element_blank()
      )
  }

  combined_chart
}

#' Add Missing Base Coefficient
#'
#' This function adds a missing base coefficient (at event time -1) if it's not present in the data.
#'
#' @param event_study_df Data frame. Event study estimates.
#'
#' @return A data frame with the base coefficient added if it was missing.
add_missing_base_coef <- function(event_study_df) {
  if (sum(event_study_df$event_time == -1, na.rm = T) == 0) {
    bind_rows(event_study_df, data.frame(type = "es_coef", event_time = -1, se = NA, beta = 0)) %>%
      arrange(event_time)
  } else event_study_df
}

#' Add Missing Base Coefficient
#'
#' This produces a number with always 'digits' number of digits. Round alone doesn't work if the number is e.g. a interger.
#'
#' @param number Numeric vector. A Vector of numbers that should be formatted
#'
#' @return A character vector formatted numbers as string
format_number <- function(number, digits = 2) {
  format(round(number, digits), nsmall = digits)
}

#' Plot Single Event Study Estimates
#'
#' This function creates a plot for a single event study estimate.
#'
#' @param event_study List or data frame. Contains event study results to be plotted.
#' @param add_supt_band Logical. Whether to add sup-t bands to the plot.
#' @param color_pre_post Logical. Whether to color pre- and post-treatment periods differently.
#' @param ribbon_instead_of_error_bar Logical. Whether to use ribbons instead of error bars for confidence intervals.
#' @param restrict_event_window Numeric vector or NULL. Limits the event window to plot.
#' @param add_tests Logical. Whether to add statistical test results to the plot.
#' @param tests_x,tests_y Numeric. X and Y coordinates for test results annotation.
#' @param tests_text_line_width,tests_font_size Numeric. Formatting for test results text.
#' @param zero_line_value Numeric or NULL. Value to display at y=0 line.
#' @param binned_endpoints Logical. Whether to use binned endpoints for x-axis labels.
#' @param manual_breaks Numeric vector or NULL. Custom x-axis breaks.
#' @param ylim Numeric vector or NULL. Y-axis limits.
#'
#' @return A ggplot object representing the event study estimates.
plot_estimates <- function(event_study,
                           add_supt_band = FALSE, # This function only works if the event_study estimates contain a suptband_upper and suptband_lower
                           color_pre_post = FALSE,
                           ribbon_instead_of_error_bar = FALSE,
                           restrict_event_window = NULL,
                           add_tests = F,
                           tests_x = 0,
                           tests_y = 0,
                           tests_text_line_width = 0.35,
                           tests_font_size = 1.75,
                           zero_line_value = NULL,
                           binned_endpoints = TRUE,
                           manual_breaks = NULL,
                           ylim = NULL
                           # column. Currently this is only produced by twfe_freyaldenhoven
) {

  # Allow to either pass a dataframe or a list containing a dataframe
  if (!is.data.frame(event_study)) {
    estimates <- event_study[["estimates"]]
  } else {
    estimates <- event_study
  }

  estimates <- event_study[["estimates"]]
  plot <- estimates %>%
    filter(type == "es_coef") %>%
    add_missing_base_coef() %>%
    mutate(
      post = ifelse(event_time > -1, "post", "pre")
    ) %>%
  {
    if (add_supt_band) {
      mutate(
        .,
        across(c(suptband_lower, suptband_upper), ~replace(.x, is.na(.x), 0))
      ) %>%
        bind_rows(
          filter(., event_time == -1) %>%
            mutate(
              post = "post"
            )
        )
    } else .
  } %>%
  {
    if (!is.null(restrict_event_window)) {
      filter(., event_time >= restrict_event_window[1], event_time <= restrict_event_window[2])
    } else .
  } %>%
    ggplot(aes(x = event_time,
               y = beta,
               color = if (color_pre_post) post else "1",
               fill = if (color_pre_post) post else NULL),
           shape = if (color_pre_post) post else NULL) +
    geom_hline(yintercept = 0, linewidth = 0.25) +
    geom_vline(xintercept = -1, linetype = "dashed", linewidth = 0.25) +
    geom_line() +
    geom_point(size = 1.5) +
    paper_theme()

  if (ribbon_instead_of_error_bar) {
    plot <- plot + geom_ribbon(aes(ymin = beta - 1.96 * se, ymax = beta + 1.96 * se), alpha = 0.3)
  } else {
    plot <- plot + geom_errorbar(aes(ymin = beta - 1.96 * se, ymax = beta + 1.96 * se), width = 0.5, alpha = 1)
  }

  if (add_supt_band) {
    if (color_pre_post) {
      plot <- plot + geom_ribbon(
        aes(ymin = suptband_lower, ymax = suptband_upper), alpha = 0.3
      )
    } else {
      plot <- plot + geom_ribbon(
        aes(ymin = suptband_lower, ymax = suptband_upper), alpha = 0.15, , color = NA, linetype = "dashed"
      )
    }
  }


  label_function_y <- function(label_numbers) {
    label_numbers %>% map_chr(function(x) {
      if (is.na(x)) {
        return(NA)
      }
      else if (x == 0 & !is.null(zero_line_value)) {
        return(str_glue("0 ({n}%)", n = format(round(zero_line_value * 100, 1), nsmall = 1)))
      } else {
        return(str_glue("{n}%", n = format(round(x * 100, 0), nsmall = 0)))
      }
    })
  }

  label_function_x <- function(label_numbers) {
    if (binned_endpoints) {
      sapply(label_numbers, function(x) {
        if (is.na(x)) {
          return(NA)
        }
        else if (x == min(label_numbers, na.rm = T)) {
          return(str_glue("\u2264{min(label_numbers)}"))
        } else if (x == max(label_numbers, na.rm = T)) {
          return(str_glue("\u2265{max(label_numbers)}"))
        } else {
          return(as.character(x))
        }
      })
    } else {
      return(as.character(label_numbers))
    }
  }

  plot <- plot + scale_y_continuous(labels = label_function_y, oob = scales::oob_keep)

  if (!is.null(ylim)) {
    plot <- plot + coord_cartesian(ylim = ylim, # This focuses the x-axis on the range of interest
                                   clip = 'off')
  }

  if (binned_endpoints) {
    plot <- plot + scale_x_continuous(
      breaks = if (is.null(manual_breaks)) seq(min(estimates$event_time), max(estimates$event_time), by = 1) else manual_breaks,
      labels = label_function_x)
  }

  if (color_pre_post) {
    plot <- plot +
      discrete_scale('shape', 'shape_d', function(n) c(21, 23, 22)[seq_len(n)], guide = "none") +
      scale_fill_discrete(guide = "none") +
      scale_color_discrete(guide = "none")
  } else {
    plot <- plot +
      scale_color_discrete(guide = "none")
  }

  if (add_tests) {
    p_parallel_trends <- test_parallel_trends(event_study[["coef"]], event_study[["vcov"]], event_study[["nobs"]]) %>% format_number()
    p_level_off <- test_level_off(event_study[["coef"]], event_study[["vcov"]], event_study[["nobs"]]) %>% format_number()

    plot <- plot + annotate(
      geom = "label",
      label = str_glue("{PARALLEL_TRENDS_TEST_STRING} = {p_parallel_trends}\n{LEVEL_OFF_TEST_STRING} = {p_level_off}"),
      x = tests_x,
      y = tests_y,
      hjust = 0,
      label.r = unit(2, "pt"),
      size = tests_font_size,
      lineheight = tests_text_line_width,
      color = "black",
      family = "XCharter Math",
      label.size = NA
    )
  }

  plot +
    labs(
      x = "Time to independence event",
      y = "Coefficient and 95% CI",
      color = "",
      fill = "",
      shape = ""
    )
}

#' Aggregate Lag Coefficients
#'
#' This function aggregates lag coefficients from an estimated model, optionally weighting by occurrence.
#'
#' @param estimated_model Object. The estimated model containing coefficients and variance-covariance matrix.
#' @param weight_by_occurnace Logical. Whether to weight coefficients by the number of observations at each lag
#' @param estimation_data Data frame or NULL. The data used for estimation, required only if weight_by_occurnace is TRUE.
#'
#' @return A tibble with aggregated coefficients, standard errors, and t-statistics for each interaction level.
aggregate_lags <- function(estimated_model, weight_by_occurnace = FALSE, estimation_data = NULL) {
  # If weight by occurance we do not equally weight coefficients but rather weight how often these lags are observed in the data
  coef <- coef(estimated_model)
  vcov <- vcov(estimated_model)

  # Order should be the same
  lag_coefcs <- tibble(
    index = which(str_detect(rownames(vcov), "^l\\d+_treating")),
    name = rownames(vcov)[str_detect(rownames(vcov), "^l\\d+_treating")]
  ) %>%
    mutate(
      interaction = str_match(name, "^l\\d+_treating:(.*)")[, 2],
      interaction = replace_na(interaction, "base")
    ) %>%
    summarise(
      indices = list(index),
      number = n(),
      .by = "interaction"
    )

  if (weight_by_occurnace) {
    max_lead <- str_extract(names(coef(estimated_model)), "(?<=d)\\d+") %>%
      as.numeric() %>%
      max(na.rm = T)
    max_lag <- str_extract(names(coef(estimated_model)), "(?<=l)\\d+") %>%
      as.numeric() %>%
      max(na.rm = T)

    # More sophisticated weighting scheme, take number of observations (ignore for now that this could vary by the interaction variable)
    weights <- estimation_data[estimated_model$obs_selection$obsRemoved,] %>%
      filter(time_post_treat != -9999) %>%
      mutate(
        time_post_treat = pmax(time_post_treat, -max_lead),
        time_post_treat = pmin(time_post_treat, max_lag)
      ) %>%
      count(time_post_treat) %>%
      filter(time_post_treat >= 0) %>%
      mutate(
        n = n / sum(n) * nrow(.) # Ensure that the average weight is 1 so that we are still dividing by the correct number when normalising by the number of coefficients later in the code
      ) %>%
      deframe()
  }

  # Constructs weights for aggreagated coefficient
  weights_matrix <- lag_coefcs %>%
    pmap(function(interaction, indices, number) {
      a <- rep(0, ncol(vcov))
      if (weight_by_occurnace) {
        a[indices] <- weights #
      } else {
        a[indices] <- 1 # Equal weights of all of the coeficients for now
      }
      a
    }) %>%
    sapply(unlist) %>%
    t()

  # This assumes that all categories estimated the same number of coefficients, which should always be the case

  # Calculate variance covariance matrix of the average (aggregate) coefficnets
  vcov_aggregate <- 1 / (lag_coefcs$number^2) * (weights_matrix %*% vcov %*% t(weights_matrix))
  coef_aggregate <- 1 / (lag_coefcs$number) * (weights_matrix %*% coef)

  # We are not quite done yet. The effect for the categories is base + cat
  # First level:
  coef_base <- coef_aggregate[1]
  interact_coef <- coef_aggregate[-1] + coef_base
  var_base <- vcov_aggregate[1, 1]

  # Calcualte variance with for loop:
  var_interact <- rep(NA, length(interact_coef))


  for (i in seq_along(var_interact)) {
    # Variance Base + Variance interaction + 2*COV
    var_interact[i] <- var_base +
      vcov_aggregate[i + 1, i + 1] +
      2 * vcov_aggregate[i + 1, 1]
  }

  lag_coefcs %>% mutate(
    coef = c(coef_base, interact_coef),
    se = sqrt(c(var_base, var_interact)),
    t_stat = coef / abs(se),
    p_value = 2 * pnorm(-abs(t_stat)) ## Assume sample is large enough so that t-stat is approximately normal.
  )
}

#' Count Observations by Cohort and Year
#'
#' This function counts the number of observations for each combination of treatment year and calendar year.
#'
#' @param speeches_data Data frame. The dataset containing speech observations.
#'
#' @return A tibble with counts of observations for each year and treatment year combination.
count_observations_by_cohort_year <- function(speeches_data) {
  speeches_data %>%
    filter(!is.na(monetary_dominance)) %>%  # Do not count missing or that will be dropped in observation
    filter(treatment_year != 0) %>% # Drop never treated
    summarise(count = n(), .by = c("year", "treatment_year")) %>%
    arrange(treatment_year, year)
}


#' Test for Parallel Trends
#'
#' This function performs a test for parallel trends in difference-in-differences models as recommended by Freyaldenhoven et al. (2021).
#'
#' @param coef Named vector. The coefficients from the estimated model.
#' @param vcov Matrix. The variance-covariance matrix of the estimated model.
#' @param nobs Numeric. The number of observations in the model.
#'
#' @return Numeric. The p-value for the parallel trends test.
test_parallel_trends <- function(coef, vcov, nobs) {
  # Note to self past self: This produces the same result as using the built fixest::wald, if only i knew of its existance before ^_^
  pre_treat_coeffcient_names <- names(coef)[str_detect(names(coef), "^d\\d+_")]

  # Subset the relevant coefficients
  beta_1 <- coef[pre_treat_coeffcient_names]
  vcov_1 <- vcov[pre_treat_coeffcient_names, pre_treat_coeffcient_names]

  # Calculate the test statistic
  q <- length(beta_1)
  F_stat <- t(beta_1) %*% solve(vcov_1) %*% beta_1 / q

  # Calculate degrees of freedom
  df1 <- q
  df2 <- nobs - length(coef)

  # Calculate the p-value
  p_value <- pf(F_stat, df1, df2, lower.tail = FALSE)[1, 1]

  return(p_value)
}

#' Test for Leveling Off
#'
#' This function performs a test for leveling off in difference-in-differences models as recommended by Freyaldenhoven et al. (2021).
#'
#' @param coef Named vector. The coefficients from the estimated model.
#' @param vcov Matrix. The variance-covariance matrix of the estimated model.
#' @param nobs Numeric. The number of observations in the model.
#'
#' @return Numeric. The p-value for the leveling off test.
test_level_off <- function(coef, vcov, nobs) {
  max_event_time <- str_extract(names(coef), "(?<=l)\\d+") %>%
    as.numeric() %>%
    max(na.rm = T)
  max_coef <- str_glue("l{max_event_time}_treating")
  second_to_max_coef <- str_glue("l{max_event_time - 1}_treating")

  # Calculate the difference between the coefficients
  beta_diff <- coef[max_coef] - coef[second_to_max_coef]

  # Extract the variance-covariance elements for the two coefficients
  var_coef1 <- vcov[max_coef, max_coef]
  var_coef2 <- vcov[second_to_max_coef, second_to_max_coef]
  cov_coef1_coef2 <- vcov[max_coef, second_to_max_coef]

  # Calculate the variance of the difference
  var_diff <- var_coef1 + var_coef2 - 2 * cov_coef1_coef2

  # Calculate the test statistic
  t_stat <- beta_diff / sqrt(var_diff)

  # Degrees of freedom
  df <- nobs - length(coef)

  # Calculate the p-value (two-tailed test). Could also use normal here prob
  p_value <- 2 * pt(abs(t_stat), df, lower.tail = FALSE)

  return(p_value)
}

