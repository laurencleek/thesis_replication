# Data loading and processing
library(arrow)
library(tidyverse)
library(slider)
library(lubridate)
library(readxl)
library(patchwork)
library(ggpattern)
library(ggsankey)

# Map distance calculations
library(geosphere)
library(sf)
library(rnaturalearth)

# Tables
library(knitr)
library(kableExtra)

source("./codes/functions/chart_themes.R")
source("./codes/functions/diff_in_diff_functions.R")
source("./codes/constants/diff_in_diff_settings.R")

source("./codes/constants/varnames.R")

# Set bounds on CBI datasets
lower_bound <- (1997 - EVENT_STUDY_NUMBER_LAGS) # Need to align this with the (max) number of leads and lags and possibly if observations are dropped
upper_bound <- 2023 #

speech_aggregation <- read_parquet("./data/processed/aggregations/speech_agg.parquet") %>%
  filter(
    year >= 1997,
    year <= 2023
  ) %>%
  mutate(
    date = as.Date(date) # Date datatype is not preserved in parquet it seems
  ) %>%
  arrange(date)

speeches_countries <- setdiff(speech_aggregation %>% pull(country) %>% unique(), "")

#---------------------------------
# Comparison of Romelli and Garriga:
#---------------------------------

# Compare country coverage in the datasets:
## Look at the raw data:
ro_indicator <- "cbie_index"
ga_indicator <- "lvau_garriga"

data_ga <- read_excel("./data/input/metadata/central_bank/CBI_2025_websiteGarriga.xlsx", sheet = "Sheet1")  %>% mutate(
        ccodewb = case_match(ccodewb, 891 ~ 688, 890 ~ 688, .default = ccodewb),
        country = countrycode(ccodewb, "iso3n", "iso3c")
) %>%
  mutate(ga_indicate = get(ga_indicator)) %>%
  select(ga_indicate, year , country)

data_ro <- read_excel("./data/input/metadata/central_bank/CBIData_Romelli_2024.xlsx", sheet = "CBI data") %>%
  select(-country) %>%
  rename(country = iso_a3) %>%
  mutate(ro_indicate = get(ro_indicator)) %>%
  select(ro_indicate, year , country)

combined_ro_ga <- expand.grid(country = speeches_countries,year = (lower_bound-1):upper_bound) %>%
    left_join(data_ga,  by = join_by(country == country, year == year)) %>%
    left_join(data_ro,  by = join_by(country == country, year == year)) %>%
  tibble() %>% arrange(
  country, year
) %>% mutate(
  ga_increase = ga_indicate > lag(ga_indicate),
  ga_decrease = (ga_indicate < lag(ga_indicate)) * -1,
  ro_increase = ro_indicate > lag(ro_indicate),
  ro_decrease = (ro_indicate < lag(ro_indicate)) * -1,
  .by = "country"
) %>%
  left_join(
    generate_treatment("ro",
                     lower_bound:upper_bound,
                     minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                     negative_strategy = "zero",
                     allow_multiple_treatments = FALSE,
                     balance_panel_strategy = "zero") %>% select(
      country, year, treating
    ) %>% mutate(
      ro_relevant_increase = treating,
      ro_relevant_decrease = (treating == -1) * -1
    ), by = join_by(country == country ,year == year)
  )  %>%
  select(-treating) %>%
  left_join(
    generate_treatment("ga",
                     lower_bound:upper_bound,
                     minimum_intensity = TREATMENT_MINIMUM_INTENSITY,
                     negative_strategy = "zero",
                     allow_multiple_treatments = FALSE,
                     balance_panel_strategy = "zero") %>% select(
      country, year, treating
    ) %>% mutate(
      ga_relevant_increase = treating,
      ga_relevant_decrease = (treating == -1) * -1
    ), by = join_by(country == country ,year == year)
  ) %>%
  select(-treating)

coverage_ro_ga <-
  combined_ro_ga %>%
    summarise(
      covered_ga = any(!is.na(ga_indicate)),
      covered_ro = any(!is.na(ro_indicate)),
      .by = country
    ) %>%
  mutate(
    coverage = case_when(
      covered_ga & covered_ro ~ "covered_both",
      covered_ro & !covered_ga ~ "covered_ro_only",
      covered_ga & !covered_ro ~ "covered_ga_only",
      !covered_ga & !covered_ro ~ "covered_none"
    )
  )

combined_ro_ga <- combined_ro_ga %>% left_join(
  coverage_ro_ga %>% select(country, coverage), by = "country"
)

# Fix strange bug where France and Norway are missing
world <- ne_countries(scale = "small", returnclass = "sf") %>% mutate(
  iso_a3 = case_when(
    iso_a3 == -99 ~ iso_a3_eh,
    TRUE ~ iso_a3
  )
)

# Fix Crimea to be part of Ukraine!
crimea <- world[world$iso_a3 == "RUS", "geometry"][[1]][[1]][[14]]
# Remove Crimea from Russia.
world[world$iso_a3 == "RUS", "geometry"][[1]][[1]][[14]] <- NULL
# Add to Ukraine
world[world$iso_a3 == "UKR", "geometry"][[1]][[1]] <- c(world[world$iso_a3 == "UKR", "geometry"][[1]][[1]], st_polygon(crimea))

dataset_comp_map <- world %>%
  filter(continent != "Antarctica") %>%
  st_transform(crs = "+proj=robin") %>%
  left_join(coverage_ro_ga, join_by(iso_a3 == country)) %>%
  ggplot() +
  geom_sf(aes(fill = coverage)) +
  #coord_sf(datum = NA) + Remove lines
  scale_fill_manual(name = "",
                    na.value = "gray90",
                    labels = c("Both datasets","Only Garriga (2025)", "No speeches"),
                    values = c("#77AFD6", "#c98d8d", "grey90"),
                    guide = guide_legend(
                      direction = "horizontal",
                      keyheight = unit(2.5, units = "mm"),
                      keywidth = unit(0.75, units = "cm"),
                      title.position = 'top',
                      title.hjust = 0.5,
                      label.hjust = 0.5,
                      nrow = 1,
                      byrow = FALSE,
                      reverse = FALSE,
                      override.aes = list(color = NA),
                      label.position = "bottom")
  ) +
  paper_theme_map()

dataset_comp_map %>% ggsave(
  filename = "./output/figures/descriptive/dataset_comp_map.pdf",
  width = panel_chart_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)