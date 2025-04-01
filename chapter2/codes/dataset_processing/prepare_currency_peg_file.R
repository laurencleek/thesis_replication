library(tidyverse)

# Extend dataset
extend_dataset <- TRUE

euro_area_composition <- list(
  "1998" = c('AUT', 'BEL', 'DEU', 'ESP', 'FIN', 'FRA', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT'),
  "2001" = c('AUT', 'BEL', 'DEU', 'ESP', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT'),
  "2007" = c('AUT', 'BEL', 'DEU', 'ESP', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SVN'),
  "2008" = c('AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LUX', 'MLT', 'NLD', 'PRT', 'SVN'),
  "2009" = c('AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LUX', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN'),
  "2011" = c('AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'EST', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LUX', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN'),
  "2014" = c('AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'EST', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LUX', 'LVA', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN'),
  "2015" = c('AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'EST', 'FIN', 'FRA', 'GRC', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN'),
  "2023" = c('AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'EST', 'FIN', 'FRA', 'GRC', 'HRV', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN')
) %>%
  enframe(name = "year", value = "members") %>%
  mutate(
    year = as.numeric(year)
  ) %>%
  complete(year = min(year):max(year)) %>%
  fill(members, .direction = "down")

is_euro_area <- function(country, year) {
  if (year < 1998) {
    return(FALSE)
  }
  country %in% (euro_area_composition[year == euro_area_composition$year, "members"] %>%
    pull() %>%
    unlist())
}

# Peg data:
peg_data <- read.csv("./data/input/metadata/country/Bilateral_DeJure_Regimes_neu.csv")

# Determine unique pegs
peg_data_proc <- peg_data %>%
  group_by(iso1, year) %>%
  summarise(pegs = list(iso2[bilateral_dejure_regime <= 3 & !is.na(bilateral_dejure_regime)]),
            currency_union = list(iso2[cu_dummy & !is.na(cu_dummy)])) %>%
  mutate(pegs = map2(pegs, currency_union, function(a, b) {
    setdiff(a, b) # Remove currency union from pegs. This should be a separate column
  }),
  ) %>%
  ungroup() %>%
  mutate(
    pegged_to = map2_chr(pegs, year, function(peg, year) {
      # Depending on which currencies are in the peg we assign the strongest to be the currency it is pegged to. In practically all
      # cases this is clear. Countries are either pegged to the Euro or the USD. Or in very few cases to a much bigger neighboring economy.
      # This approach with assigning the pegged to country could go wrong if the country that was pegged to previously was pegged to another
      # economy. E.g. if india would have ever been pegged to the US dollar, it would assign the US as pegged to india.
      # Manully checking seems that this is not a problem.
      if ("USA" %in% peg) {
        return("USA")
      }
      else if (any(peg %>% map_lgl(~is_euro_area(.x, year)))) {
        return("EA")
      }
      else if ("IND" %in% peg) {
        return("IND")
      }
      else if ("AUS" %in% peg) {
        return("AUS")
      }
      else if ("ZAF" %in% peg) {
        return("ZAF")
      }
      else if ("SGP" %in% peg) {
        return("SGP")
      }
      else {
        return(NA)
      }
    })) %>%
  mutate(
    pegged = !is.na(pegged_to),
    member_of_currency_union = map_lgl(currency_union, ~length(.x) > 0)
  ) %>%
  rename(
    country = "iso1"
  )

# The euro area is the only non-pegged curreny area in teh dataset
if (extend_dataset) {
  peg_data_proc <- peg_data_proc %>%
    group_by(country) %>%
    complete(
      year = 1997:2023
    ) %>%
    ungroup() %>%
    mutate(
      euro_area = map2_lgl(country, year, ~is_euro_area(.x, .y))
    ) %>%
    mutate(
      member_of_currency_union =
        case_when(
          country %in% c('AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'EST', 'FIN', 'FRA', 'GRC', 'HRV', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN') ~ euro_area,
          TRUE ~ member_of_currency_union
        )
    ) %>%
    group_by(country) %>%
    fill(
      pegged,
      member_of_currency_union,
      .direction = "updown"
    )
}

peg_data_proc %>%
  select(country, year, pegged, member_of_currency_union) %>%
  write_csv("./data/processed/currency_pegs.csv")