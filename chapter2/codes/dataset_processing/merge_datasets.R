# This R code produces aggregations of the dominance dataset and merges for different purposes

# Process data
library(tidyverse)
library(readxl)
library(haven)
library(lubridate)
library(zoo)
library(arrow)
library(countrycode)

# A reduced metadata list for a smaller dataset
reduced_metadata <- c("country", "currency", "currency_code", "democracy_ind", "advanced", "gdp_real_ppp_capita", "ro_diff_cbie_index", "ro_cbie_index", "financial_stress", "cbis_res", "cbis")

# Metadata speech aggreation
metadata_speech_agg <- c("country", "date", "currency", "currency_code",
                         "diff_unemployment", "unemployment_rate", "diff_inflation", "inflation", "democracy_ind",
                         "advanced", "government_debt_pgdp", "structural_balance_pgdp", "gdp_real_ppp_capita",
                         "speaker", "occasion", "venue", "output_gap_pgdp",
                         "gdp_real_growth", "interst_payments_p_gdp", "inflation_off_target", "i_minus_g", "gdp_deflator_pct",
                         "location", "latitude", "longitude", "audience",
                         "ro_diff_cbie_index", "ro_cbie_index", "ro_cbie_obj", "ro_cbie_policy_q3",
                         "gdp_real_growth", "financial_stress", "cbis_res", "backward_looking", "forward_looking", "current", "cbis", "imp_interest", "output_gap_pgdp",
                         "cb_location", "cb_longitude", "cb_latitude", "independence_judiciary", "judiciary_purges", "judiciary_attacks", "pegged", "member_of_currency_union"
)

metadata_central_bank_year_agg <- c("country", "currency", "currency_code", "diff_unemployment", "unemployment_rate", "diff_inflation", "inflation", "democracy_ind",
                                    "advanced", "government_debt_pgdp", "structural_balance_pgdp", "gdp_real_ppp_capita", "output_gap_pgdp",
                                    "ro_diff_cbie_index", "ro_cbie_index", "ro_cbie_obj", "ro_cbie_policy_q3",
                                    "gdp_real_growth", "primary_balance_pgdp",
                                    "gdp_real_growth", "interst_payments_p_gdp", "inflation_off_target", "i_minus_g", "imp_interest",
                                    "financial_stress", "cbis_res", "cbis", "gdp_deflator_pct",
                                    "cb_location", "cb_longitude", "cb_latitude", "independence_judiciary", "judiciary_purges", "judiciary_attacks", "pegged", "member_of_currency_union"
)

metadata_country_year_agg <- metadata_central_bank_year_agg

clean_column_names <- function(column_name) {
  column_name %>%
    tolower() %>%
    str_replace("-", "_") %>%
    str_replace("\\s+", "_")
}

one_hot_encoding <- function(df, column_name, prefix = "") {
  df %>%
    mutate(value = 1) %>%
  {
    if (prefix != "") {
      mutate(., "{column_name}" := paste(prefix, !!sym(column_name), sep = "_"))
    } else { . }
  } %>%
    pivot_wider(names_from = column_name, values_from = value, values_fill = 0) %>%
  {
    if (prefix != "") {
      select(., -one_of(str_glue("{column_name}_NA")))
    } else {
      select(., -one_of(str_glue("NA")))
    }
  }
}

# This is a experimental function to smooth outputs
smooth_series <- function(series) {
  share_missig <- sum(is.na(series)) / length(series)

  span = min(1 / (1 - share_missig), 2) * 0.4

  if (sum(!is.na(series)) < 3) {
    return(series)
  }
  smoothed_data <- series
  smoothed_data[!is.na(series)] <- loess(series ~ seq_along(series), data = data.frame(series = series), span = 0.4)$fitted
  smoothed_data <- na.approx(zoo(smoothed_data), na.rm = F) %>% as.vector()
}

aggregate_sentence_level_data <- function(
  aggregation_mode = "speech", # One of "quarter", "year", "central_bank_year", "speech"
  smoothing = FALSE, # Whether to apply smoothing
  calculate_dominances = TRUE, # If this is set to false, it will not calculate the relative indicator
  main_dominance_indicator = "_rel_domcorp", # This indicator get's renamed to e.g. "monetary_dominance" without the _ indicating which indicator it is
  additional_dominance_indicators = NULL, # Optional list of additional indicators to include. Only relevant when main_dominance_indicator is not NULL.
  metadata_of_interest = NULL, # If set to null will output all metadata
  suffix = NULL, # Suffix to save file as
  fill_missing_country_year_combinations = FALSE, # If TRUE country, year observations for which there are no speeches are added with missing NA in the dominance columns. Useful for running regressions with the metadata variables
  skip_stata = FALSE # Stata doesnt like long variable names. If true doesnt save as Stata
) {

  # Determine grouping variables based on how the aggregation should work
  if (aggregation_mode == "year") {
    grouping_vars <- "year"
  } else if (aggregation_mode == "central_bank_year") {
    grouping_vars <- c("year", "central_bank")
  } else if (aggregation_mode == "speech") {
    grouping_vars <- c("year", "speech_identifier", "central_bank")
  } else if (aggregation_mode == "country_year") {
    grouping_vars <- c("year", "country")
  }

  # Sentence level data
  sentences_bis <- read_parquet(str_glue("./data/processed/sentence_ids_speeches.parquet")) %>%
    select(sentence_id, speech_identifier) # Only keep relevant for aggregaton

  classification <- read_parquet(str_glue("./data/processed/dom_corp_classification_sentence_level.parquet"))
  dominance_classifications <- clean_column_names(setdiff(unique(classification$classification), "NA"))

  imf_vars <- list(
    "NGDP" = "gdp_nominal",
    "NGDPPC" = "gdp_nominal_capita",
    "NGDPD" = "gdp_nominal_usd",
    "NGDPDPC" = "gdp_nominal_usd_capita",
    "NGDP_R" = "gdp_real", # National currency
    "NGDPRPC" = "gdp_real_capita", # National currency
    "NGDPRPPPPC" = "gdp_real_ppp_capita", # PPP USD
    "GGEI" = "interest_payments",
    "GGXWDG" = "government_debt",
    "GGXONLB" = "primary_balance",
    "GGXCNL" = "budget_balance",
    "GGX" = "total_expenditure",
    "GGR" = "total_revenue",
    "GGSB" = "structural_balance",
    "GGXCNL_NGDP" = "primary_balance_pgdp",
    "GGXWDG_NGDP" = "government_debt_pgdp",
    "GGXCNL_NGDP" = "budget_balance_pdgp",
    "GGR_NGDP" = "total_revene_pgdp",
    "GGX_NGDP" = "total_expenditure_pgdp",
    "GGSB_NPGDP" = "structural_balance_pgdp",
    "NGAP_R" = "output_gap",
    "NGAP_NPGDP" = "output_gap_pgdp",
    "PCPI" = "hicp",
    "NGDP_D" = "gdp_deflator",
    "LUR" = "unemployment_rate",
    "LE" = "employment",
    "LP" = "population"
  )

  # Download WEO the paths follow this pattern https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2023/WEOApr2023all.xls'
  # And then save as xlsx and replace all "(n/a)" with "" before saving.
  weo <- read_excel("./data/input/metadata/country/WEOApr2024all.xlsx") %>%
    mutate(
      across(matches("\\d{4}"), ~as.numeric(.x))
    ) %>%
    select(ISO, `WEO Subject Code`, matches("\\d{4}")) %>%
    rename(country = "ISO", imf_var = "WEO Subject Code") %>%
    filter(imf_var %in% names(imf_vars)) %>%
    mutate(
      imf_var = dplyr::recode(imf_var, !!!imf_vars)
    ) %>%
    pivot_longer(
      cols = matches("\\d{4}"),
      names_to = "year",
      values_to = "value"
    ) %>%
    pivot_wider(
      values_from = "value",
      names_from = "imf_var"
    ) %>%
    mutate(
      year = as.numeric(year),
    ) %>%
    mutate(
      inflation = (hicp - lag(hicp)) / lag(hicp),
      gdp_deflator_pct = (gdp_deflator - lag(gdp_deflator)) / lag(gdp_deflator),
      gdp_real_growth = (gdp_real - lag(gdp_real)) / lag(gdp_real),
      gdp_nominal_growth = (gdp_nominal - lag(gdp_nominal)) / lag(gdp_nominal),
      interst_payments = primary_balance - budget_balance,
      interst_payments_p_gdp = interst_payments / gdp_nominal,
      imp_interest = interst_payments / lag(government_debt),
      i_minus_g = imp_interest - gdp_nominal_growth,
      .by = "country"
    ) %>%
    mutate(
      global_unemployment = weighted.mean(unemployment_rate, coalesce(gdp_nominal_usd, 0), na.rm = T),
      global_inflation = weighted.mean(pmin(inflation, 1), coalesce(gdp_nominal_usd, 0), na.rm = T),
      global_gdp_nominal_growth = weighted.mean(gdp_nominal_growth, coalesce(gdp_nominal_usd, 0), na.rm = T),
      .by = "year"
    ) %>%
    mutate(
      diff_inflation = inflation - lag(inflation),
      inflation_off_target = inflation - 0.02,
      diff_unemployment = unemployment_rate - lag(unemployment_rate),
      .by = "country"
    )

  # Financial stress indicator IMF:
  financial_stress <- read_excel("./data/input/metadata/country/FSI_Dataset.xlsx", sheet = "T1") %>%
    mutate(
      year = as.numeric(str_extract(year, "\\d{4}"))
    ) %>%
    summarise(
      across(everything(), ~mean(.x)), .by = "year"
    ) %>%
    pivot_longer(-year, names_to = "country", values_to = "financial_stress")

  

  # Central bank level metadata
  country_and_currencies <- read.csv("./data/input/metadata/central_bank/currencies_country.csv") %>% mutate(
    euro_area = currency_code == "EUR",
    dollar = currency_code == "USD"
  )

  currency_pegs <- read.csv("./data/processed/currency_pegs.csv") %>% tibble()

  speech_level_metadata <- read_parquet("./data/processed/complete_metadata.parquet")

  

  independence_romelli <- read_excel("./data/input/metadata/central_bank/CBIData_Romelli_2024.xlsx", sheet = "CBI data") %>%
    select(iso_a3, year, all_of((c("cbie_index", "cbie_board", "cbie_policy", "cbie_obj", "cbie_lending", "cbie_finindep", "cbie_report", "cbie_policy_q3")))) %>%
    mutate(
      across(all_of(c("cbie_index", "cbie_board", "cbie_policy", "cbie_obj", "cbie_lending", "cbie_finindep", "cbie_report")), ~.x - lag(.x), .names = "diff_{col}"),
      across(all_of(c("cbie_index", "cbie_board", "cbie_policy", "cbie_obj", "cbie_lending", "cbie_finindep", "cbie_report")), ~(.x - lag(.x)) / lag(.x), .names = "pct_{col}"),
      .by = "iso_a3"
    ) %>%
    rename_with(~paste("ro", { .x }, sep = "_"), .cols = -c("iso_a3", "year"))

  

  # Supervision (Only until 2013)
  romelli_cbis <- read_stata("./data/input/metadata/central_bank/CBISIndex.dta") %>%
    rename("cbis" = "CBISIndex", "cbis_res" = "CBISBisIndex", "country" = "iso_a3") %>%
    select(country, year, cbis, cbis_res)

  # Covers all counries in the speeches set except Macau and Bahamas
  vdem <- read_parquet("./data/processed/vdem_clean.parquet") %>%
    mutate(
      democracy_ind = case_when(
        v2x_regime == 0 ~ 0,
        v2x_regime == 1 ~ 0,
        v2x_regime == 2 ~ 1,
        v2x_regime == 3 ~ 1,
        TRUE ~ NA),
      independence_judiciary = (v2jupoatck + v2jupurge) / 2,
      judiciary_attacks = v2jupoatck,
      judiciary_purges = v2jupurge
    ) %>%
    bind_rows(
      tibble(country = "EA", year = 1998:2023, democracy_ind = 1) # Assign EURO area as democracy
    )

  # END LOADING DATASETS
  full_sentence_level_classification <- sentences_bis %>%
    left_join(speech_level_metadata %>% select(date, central_bank, speech_identifier), by = "speech_identifier") %>%
    mutate(
      date = as.Date(date)
    ) %>%
    relocate(speech_identifier, date) %>%
    left_join(
      classification %>% rename("classification" = classification),
      by = join_by(sentence_id)
    ) %>%
    rename_with(
      clean_column_names
    )

  aggregation_data <- full_sentence_level_classification %>%
    left_join(
      country_and_currencies %>% select(central_bank, country),
      by = join_by(central_bank == central_bank)) %>%
    mutate(
      year = year(floor_date(date, "year")),
      quarter = quarter(floor_date(date, "quarter"))
    ) %>%
    one_hot_encoding("classification") %>%
    rename_with(
      clean_column_names
    ) %>%
    summarise(
      across(
        all_of(c(dominance_classifications)),
        ~sum(.x, na.rm = TRUE)
      ),
      number_of_sentences = n(),
      number_of_speeches = length(unique(speech_identifier)),
      .by = grouping_vars
    ) %>%
  {
    if (smoothing) {
      mutate(
        across(
          all_of(c(l1_levels, l2_levels, dominance_classifications)),
          ~smooth_series(.x),
          .names = "s{.col}"
        )
      )
    } else .
  }

  if (calculate_dominances) {
    aggregation_data <- aggregation_data %>%
      mutate(
        fiscal_dom_or_corp = fiscal_dominance + monetary_fiscal_coordination,
        financial_dom_or_corp = financial_dominance + monetary_financial_coordination,
        across(
          all_of(dominance_classifications),
          ~.x,
          .names = "{.col}_abs"
        ),
        across(
          all_of(c(dominance_classifications, "fiscal_dom_or_corp", "financial_dom_or_corp")),
          ~.x / number_of_sentences,
          .names = "{.col}_rel_sentence"
        ),
        across(
          all_of(c(dominance_classifications, "fiscal_dom_or_corp", "financial_dom_or_corp")),
          ~.x / number_of_speeches,
          .names = "{.col}_rel_speech"
        ),
        across(
          all_of(c("monetary_dominance", "financial_dominance", "fiscal_dominance")),
          ~.x / (monetary_dominance +
            financial_dominance +
            fiscal_dominance),
          .names = "{.col}_rel_dom"
        ),
        across(
          all_of(c("monetary_dominance", "monetary_financial_coordination", "monetary_fiscal_coordination")),
          ~.x / (monetary_financial_coordination +
            monetary_dominance +
            monetary_fiscal_coordination),
          .names = "{.col}_rel_corp"
        ),
        across(
          all_of(c("monetary_dominance", "financial_dominance", "fiscal_dominance", "monetary_fiscal_coordination", "monetary_financial_coordination", "financial_dom_or_corp", "fiscal_dom_or_corp")),
          ~.x / (monetary_dominance +
            financial_dominance +
            fiscal_dominance +
            monetary_financial_coordination +
            monetary_fiscal_coordination),
          .names = "{.col}_rel_domcorp"
        ),
      )
  }

  if (calculate_dominances & !is.null(main_dominance_indicator)) {
    aggregation_data <- aggregation_data %>%
      select(
        all_of(grouping_vars),
        ends_with(main_dominance_indicator),
        ends_with(if (is.null(additional_dominance_indicators)) "__________________" else additional_dominance_indicators), # If null put a random string that ideally matches nothing
        number_of_speeches, number_of_sentences
      ) %>%
      rename_with(~str_replace(.x, main_dominance_indicator, "")) # Selecting and renaming like this is a bit risky if there are other variables that follow the same pattern
  }

  if (fill_missing_country_year_combinations) {
    if ("central_bank" %in% grouping_vars) {
      aggregation_data <- aggregation_data %>%
        complete(
          central_bank, year
        )
    } else if ("country" %in% grouping_vars) {
      aggregation_data <- aggregation_data %>%
        complete(
          country, year
        )
    }
  }

  # Merge metadata depending on grouping
  aggregation_data_with_metadata <- aggregation_data %>%
  {
    if ("central_bank" %in% grouping_vars) {
      left_join(., country_and_currencies, by = join_by(central_bank == central_bank))
    } else if ("country" %in% grouping_vars) { # Making the assumption that we do not have central bank and country at the same time in grouping which wouldn't make sense
      left_join(., country_and_currencies %>% summarise(across(-central_bank, ~.x[1]), .by = "country"), by = join_by(country == country))
    }
  } %>%
  {
    if (("central_bank" %in% grouping_vars | "country" %in% grouping_vars) & "year" %in% grouping_vars) {
      left_join(., independence_romelli, by = join_by(country == iso_a3, year == year)) %>%
        left_join(weo, na_matches = "never", by = join_by(country == country, year == year)) %>%
        left_join(vdem, na_matches = "never", by = join_by(country == country, year == year)) %>%
        left_join(financial_stress, na_matches = "never", by = join_by(country == country, year == year)) %>%
        left_join(romelli_cbis, na_matches = "never", by = join_by(country == country, year == year)) %>%
        
        left_join(currency_pegs, na_matches = "never", by = join_by(country == country, year == year))
    } else .
  } %>%
  {
    if ("speech_identifier" %in% grouping_vars) {
      left_join(., speech_level_metadata %>% select(-central_bank), by = join_by(speech_identifier == speech_identifier))
       
    } else .
  } %>%
    rename_with(
      clean_column_names
    )

  if (!is.null(metadata_of_interest)) {
    aggregation_data_with_metadata <- aggregation_data_with_metadata %>% select(
      any_of(colnames(aggregation_data)), any_of(metadata_of_interest))
  }

  # If suffix not provided
  if (is.null(suffix)) {
    suffix <- ifelse(is.null(metadata_of_interest), "all", "")
  }

  # Save Stata
  if (!skip_stata) {
    aggregation_data_with_metadata %>% write_dta(str_glue("./data/processed/aggregations/{aggregation_mode}_agg{suffix}.dta"))
  }

  # Save Parquet
  aggregation_data_with_metadata %>% write_parquet(str_glue("./data/processed/aggregations/{aggregation_mode}_agg{suffix}.parquet"))

  # Save csv
  aggregation_data_with_metadata %>% write_csv(str_glue("./data/processed/aggregations/{aggregation_mode}_agg{suffix}.csv"))
}

aggregate_sentence_level_data(
  aggregation_mode = "speech", # One of "quarter", "year", "central_bank_year", "speech"
  smoothing = FALSE, # Whether to apply smoothing
  calculate_dominances = TRUE, # If this is set to false, it will not calculate the relative indicator
  main_dominance_indicator = "_rel_domcorp", # If set to not null drops all other than choosen indicator and renames to fiscal_dominance, monetary dominance etc..
    metadata_of_interest = metadata_speech_agg,
  skip_stata = TRUE
)

aggregate_sentence_level_data(
  aggregation_mode = "central_bank_year", # One of "quarter", "year", "central_bank_year", "speech"
  smoothing = FALSE, # Whether to apply smoothing
  calculate_dominances = TRUE, # If this is set to false, it will not calculate the relative indicator
  main_dominance_indicator = "_rel_domcorp", # If set to not null drops all other than choosen indicator and renames to fiscal_dominance, monetary dominance etc..
  metadata_of_interest = metadata_central_bank_year_agg,
  skip_stata = TRUE
)

aggregate_sentence_level_data(
  aggregation_mode = "country_year", # One of "quarter", "year", "central_bank_year", "speech"
  smoothing = FALSE, # Whether to apply smoothing
  calculate_dominances = TRUE, # If this is set to false, it will not calculate the relative indicator
  main_dominance_indicator = "_rel_domcorp", # If set to not null drops all other than choosen indicator and renames to fiscal_dominance, monetary dominance etc..
  metadata_of_interest = metadata_country_year_agg,
  fill_missing_country_year_combinations = TRUE,
  skip_stata = TRUE
)