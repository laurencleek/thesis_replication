# Generate VDEM data
library(tidyverse)
library(vdemdata)
library(arrow)
library(countrycode)

vdem_code_book <- vdemdata::codebook %>%
  tibble() %>%
  select(name, tag, scale, responses) %>%
  mutate(
    clean_scale = case_when(
      str_detect(scale, "converted to interval") ~ "interval",
      str_detect(scale, "^Interval") ~ "interval",
      str_detect(scale, "^Numeric") ~ "interval",
      str_detect(scale, "^Continuous") ~ "interval",
      str_detect(scale, "^Ordinal") ~ "ordinal",
      str_detect(scale, "^Dichotomous") ~ "binary",
      str_detect(scale, "^yes/no") ~ "interval", # There is one variable which is actually on a interval scale
      str_detect(scale, "^Nominal") ~ "nominal",
      str_detect(scale, "^Series of dichotomous scales") ~ "nominal",
      str_detect(scale, "^We provide two versions of this index") ~ "interval"
    )
  )

# Load VDEM Dataset
vdem_dataset <- vdemdata::vdem %>% tibble()

to_drop <- c("v2regidnr", # An identifier which identifies regimes.
             "v3lgcamoth", # Identifies chambeer names which are very different in each country (not useful in explaining cross sectiona variation)
             "country_text_id", # Country name abbreviation
             "v2zz", # Question about coder background
             "v3zz", # Question about coder background historical
             "project", # Vdem project
             "comment", # Drop comments

             # Coding meta data
             "codingstart",
             "codingend_contemp",
             "codingend",
             "gapend",
             "gapindex",
             "historical_date",

             "v2histname", #Time specific country name
             "e_democracy_omitteddata", # Democracy indicator with different missing values
             "e_gdp", # GDP is in WEO
             "e_gdppc", # GDP is in WEO
             "e_miinflat" # Inflation in WEO
)

comb_regex <- function(regexes) {
  return(
    str_glue("({innerpart})", innerpart = paste0(regexes, collapse = "|"))
  )
}

# Remove variables that we are dropping
vdem_code_book <- vdem_code_book %>% filter(
  !str_detect(tag, comb_regex(to_drop))
)

na_scale_var_names <- vdem_code_book %>%
  filter(is.na(scale)) %>%
  pull(tag)
na_scale_vars <- vdem_dataset %>% select(any_of(na_scale_var_names))

manually_classify <- FALSE
if (manually_classify) {
  scales <- c()
  for (name in names(na_scale_vars)) {
    message(
      str_glue("Name: {var_info(name)[['name']]} \n Quest: {var_info(name)[['question']]} \n  Res: n{var_info(name)[['responses']]}"))
    na_scale_vars %>%
      count(get(name)) %>%
      print(n = 5)
    scale <- readline(prompt = "Enter scale, i, b, o, n, d")
    scales <- c(scales, scale)
  }
  tibble(var_names = names(na_scale_vars), scale = scales) %>%
    write.csv("../../data/processed/lasso/vdem_manual_scales.csv", row.names = F)
}

# Consider variables from the codebook that are not in the data and see if they are encoded already maybe.
variables_that_are_somehow_matched <- vdem_dataset %>%
  select(
    matches(comb_regex(setdiff(na_scale_var_names, names(na_scale_vars))))
  ) %>%
  names()
# Conclusion: -> missing variables are comments or coding metadata that is not of interest

# Load manually assigned scales
manually_assigned_scale <-
  read.csv("../../data/processed/lasso/vdem_manual_scales.csv") %>%
    filter(scale != "d") %>%  # Some are marked with d for delete, this includes text columns (like name of head of state, and some reduncies like differently coded geographic regions) or clearly irrelevant variables (year in which HoS was born)
    mutate(scale = case_match(scale,
                              "o" ~ "ordinal",
                              "i" ~ "interval",
                              "n" ~ "nominal",
                              "b" ~ "binary"
    )) %>%
    rename(tag = "var_names", clean_scale = scale)

vdem_code_book <- vdem_code_book %>%
  rows_update(
    manually_assigned_scale, by = "tag"
  )

# Clean e_democ, e_autoc, e_polity2, e_polcomp, e_p_polity
vdem_dataset <- vdem_dataset %>%
  mutate(
    across(all_of(c("e_p_polity", "e_democ", "e_autoc", "e_polity2", "e_polcomp")), ~case_when(.x == -66 ~ NA,
                                                                                               .x == -77 ~ 0,
                                                                                               .x == -88 ~ NA,
                                                                                               TRUE ~ .x
    ))
  )

one_hot_encoder <- function(df, columns = "all", drop_first_column = T, na_strategy = NA, remove_column = T, encode_as_numeric = F) {
  if (columns == "all") columns = names(df)

  one_hot_columns <- columns %>% map_dfc(function(column) {
    df %>%
      select(all_of(column)) %>%
    {
      if (encode_as_numeric) {
        mutate(., "{column}" := match(get(column), unique(get(column))))
      } else .
    } %>%
      mutate(
        value = 1, id = row_number()
      ) %>%
      pivot_wider(names_from = column, values_fill = 0) %>%
      select(-id) %>%
      rename_with(~str_glue("{column}_{.x}")) %>%
    {
      if (is.na(na_strategy) & sum(is.na(df[[column]])) > 0) {
        # If there is at least one na and the na strategy is na, we replace all with na
        .[is.na(df[[column]]),] <- NA
        select(., -all_of(str_glue("{column}_NA")))
      } else if (na_strategy == "zero" & sum(is.na(df[[column]])) > 0) {
        # If it zero, we justdrop the NA column
        select(., -all_of(str_glue("{column}_NA")))
      } else .
    } %>%
    {
      # We might need a reference category to prevent collinearity and therefore drop the first column
      if (drop_first_column) {
        .[, -1]
      } else .
    }
  })

  if (remove_column) {
    return(
      bind_cols(
        df %>% select(-all_of(columns)),
        one_hot_columns
      )
    )
  } else {
    return(
      bind_cols(
        df,
        one_hot_columns
      )
    )
  }
}

# Nominal columns directly included:
nominal <- vdem_code_book %>%
  filter(clean_scale == "nominal") %>%
  pull(tag)

# Numerically encoded
numcoded_nominal <- vdem_dataset %>% select(any_of(nominal))

# Check if actually nominally encoded by looking at the number of unique values. There are only a couple with more than 5
numcoded_nominal %>%
  summarise(across(everything(), ~length(unique(.x[!is.na(.x)])))) %>%
  select(where(~sum(.x) > 5))

# Onehot encode these variables
onehot_numcoded_nominal <- numcoded_nominal %>%
  one_hot_encoder(drop_first_column = FALSE, encode_as_numeric = F)

# Onehot encoded nominal
onehot_nominal <- vdem_dataset %>% select(matches(str_glue("({regex})", regex = paste(paste0(nominal, "_\\d+"), collapse = "|"))))

nominal_data <- bind_cols(onehot_numcoded_nominal, onehot_nominal)

# Is there overlap? So far no!
intersect(names(numcoded_nominal), unique(str_replace(names(onehot_nominal), "_\\d+$", "")))

# Ordinal variables:
ordinal <- vdem_code_book %>%
  filter(clean_scale == "ordinal") %>%
  pull(tag)

# Some ordinal variables are just identified by their variable name
single_spec_ordinal <- vdem_dataset %>% select(any_of(ordinal))

# Some indicators are available on scales of multiple resolution; use highest "5C"
multi_spec_ordinal_5c <- vdem_dataset %>% select(matches(str_glue("({regex})", regex = paste(paste0(ordinal, "_5C"), collapse = "|"))))

ordinal_data <- bind_cols(single_spec_ordinal, multi_spec_ordinal_5c)

# Test for overlap. Should be zero or we are inclduing the same variables twice.
intersect(names(single_spec_ordinal), unique(str_replace(names(multi_spec_ordinal_5c), "_5C", "")))

# Interval variables:
interval <- vdem_code_book %>%
  filter(clean_scale == "interval") %>%
  pull(tag)

interval_data <- vdem_dataset %>% select(any_of(interval))

# Binary variables
binary <- vdem_code_book %>%
  filter(clean_scale == "binary") %>%
  pull(tag)

# Test if variables are really binary. (Some variables are binary but aggregated into an index
# that seems to incorporate coder uncertainty like )
not_binary <- vdem_dataset %>%
  select(any_of(binary)) %>%
  summarise(across(everything(), ~length(unique(.x[!is.na(.x)])))) %>%
  select(where(~sum(.x) > 2)) %>%
  names()

binary_data <- vdem_dataset %>% select(all_of(binary))

# Variables not being binary is no problem but we should mark them as not binary in the codebook:
vdem_code_book <- vdem_code_book %>%
  mutate(
    clean_scale = case_when(
      tag %in% not_binary ~ "interval",
      TRUE ~ clean_scale
    )
  )

combined_and_cleaned_vdem <- bind_cols(
  vdem_dataset %>% select(year, country_name),
  nominal_data,
  ordinal_data,
  interval_data,
  binary_data,
)

# Clean variables names that sould match the tag column of the codebook
clean_nammes <- combined_and_cleaned_vdem %>%
  names() %>%
  str_replace("_\\d+\\.\\d+$", "") %>%
  str_replace("_\\d+$", "") %>%
  str_replace("_5C$", "") %>%
  unique()

# Not in dataset (because dropped, textual scale or actually not contained) Around 70 variables
setdiff(vdem_code_book$tag, clean_nammes)

# In codebook but not in dataset
in_codebook_but_not_dataset <- setdiff(vdem_code_book$tag, clean_nammes)

# Relevant codesbook vars:
vdem_code_book %>%
  filter(
    tag %in% clean_nammes
  ) %>%
  write_parquet("../../data/processed/vdem_codebook.parquet")

combined_and_cleaned_vdem %>%
  mutate(
    country = countrycode(country_name, "country.name", "iso3c")
  ) %>%
  relocate(country, year) %>%
  select(-country_name) %>%
  filter(!is.na(country)) %>%
  write_parquet("../../data/processed/vdem_clean.parquet")