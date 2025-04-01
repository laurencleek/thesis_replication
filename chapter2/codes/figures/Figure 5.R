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

# Sentence level Figure:
sentence_level_data <- read_parquet(str_glue("./data/processed/dom_corp_classification_sentence_level.parquet"))

# Prepare the data for the Sankey diagram
classifications_sankey <- sentence_level_data %>%
  group_by(classification) %>%
  summarise(Count = n()) %>%
  mutate(Percentage = Count / sum(Count) * 100) %>%
  mutate(Origin = "All sentences") %>%
  select(Origin, classification, Count, Percentage) %>%
  mutate(
    classification = str_to_title(classification)
  )

classifications_sankey_long <- classifications_sankey %>%
  make_long(Origin, classification, value = Count) %>%
  left_join(classifications_sankey, by = c("node" = "classification")) %>%
  mutate(
    Count = case_when(
      is.na(Count) ~ sum(Count, na.rm = T),
      TRUE ~ Count
    )
  ) %>%
  mutate(Label = ifelse(
    is.na(Percentage),
    paste0(node, "\n n = ", scales::comma(Count)),
    paste0(node, "\n n = ", scales::comma(Count), " (", round(Percentage, 1), "%)")
  )) %>%
  mutate(node_color = as.numeric(as.factor(node))) # Assign numeric IDs for grayscale differentiation

classifications_sankey_long <- classifications_sankey_long %>% mutate(
  across(c(node, next_node), ~ as.factor(.x))
) %>%
  mutate(
    node = factor(node, levels = c(levels(node), " ", "  ")) # Add empty levels for padding
  )

classifications_plot <- ggplot(classifications_sankey_long, aes(
  x = x, next_x = next_x, node = node, next_node = next_node, value = value
)) +
  geom_sankey(
    aes(fill = node_color), # Use grayscale mapping
    flow.alpha = 0.2, width = 0.01 # Thinner lines for flows
  ) +
  scale_fill_gradient(low = "gray20", high = "gray90") + # Inverted shades of gray for nodes
  scale_color_gradient(low = "gray20", high = "gray90") + # Inverted shades of gray for edges
  geom_sankey_label(aes(
    label = Label,
    family = "XCharter Math",
    y = node,
    x = ifelse(x == "Origin", as.numeric(x) - 0.20, as.numeric(x) + 0.02) # Adjust label alignment
  ), hjust = 0, size = 2.2, color = "black") +
  theme_void() +
  theme(
    legend.position = "none",
    plot.margin = margin(10, 50, 10, 6) # Margins if needed
  ) +
  labs(title = NULL, x = NULL, y = NULL) +
  scale_x_discrete(expand = expansion(mult = c(-0.25, 0.25))) +
  coord_cartesian(clip = 'off') # Ensure clipping is disabled

# Save the plot
ggsave(
  classifications_plot,
  filename = "output/figures/descriptive/classifications.pdf",
  width = panel_chart_width,
  height = default_chart_height,
  device = cairo_pdf,
  units = "cm"
)