library(arrow)
library(tidyverse)
library(patchwork)
library(ggbeeswarm)

source("./codes/functions/chart_themes.R")

trained_models_scatter <- read_parquet("./data/processed/fine_tuning/study_115.parquet") %>%
  select(starts_with("user_attrs")) %>%
  rename_with(~str_replace(.x, "user_attrs_", "")) %>%
  filter(!is.na(f1_macro), f1_macro != 0) %>% # Drop failed runs
  select(all_of(c("f1_macro", "temperature", "batch_size", "upsample_factor"))) %>%
  pivot_longer(-f1_macro) %>%
  mutate(
    name = case_match(name,
                      "batch_size" ~ "Batch size",
                      "temperature" ~ "Temperature",
                      "upsample_factor" ~ "Upsample factor",
    )
  ) %>%
  ggplot(aes(x = value, y = f1_macro, color = name)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = 'lm', se = F) +
  facet_wrap(~name, scales = "free") +
  paper_theme() +
  labs(
    x = "",
    y = "F1 macro score"
  ) +
  guides(color = "none")

trained_models_swarms <- read_parquet("./data/processed/fine_tuning/study_115.parquet") %>%
  select(starts_with("user_attrs"), starts_with("params_add_synthetic")) %>%
  rename_with(~str_replace(.x, "user_attrs_", "")) %>%
  filter(!is.na(f1_macro), f1_macro != 0) %>% # Drop failed runs
  select(all_of(c("f1_macro", "train_classify_in_prompt", "params_add_synthetic", "randomized_epochs"))) %>%
  mutate(
    train_classify_in_prompt = as.character(train_classify_in_prompt),
    params_add_synthetic = case_match(
      params_add_synthetic,
      FALSE ~ "no",
      TRUE ~ "yes"
    ),
    randomized_epochs = case_match(
      randomized_epochs,
      FALSE ~ "no",
      TRUE ~ "yes"
    )) %>%
  pivot_longer(-f1_macro) %>%
  mutate(
    name = case_match(name,
                      "train_classify_in_prompt" ~ "Sentences per prompt",
                      "params_add_synthetic" ~ "Add synthetic examples",
                      "randomized_epochs" ~ "Randomize epochs"

    ),
    value = factor(value, levels = c('no', 'yes', '5', '10', '25'))
  ) %>%
  ggplot(aes(x = value, y = f1_macro, color = name)) +
  geom_quasirandom(alpha = 0.5) +
  stat_summary(
    aes(fill = name),
    geom = "tile",
    fun.y = "mean",
    width = 0.6,
    show.legend = FALSE
  ) +
  stat_summary(
    aes(fill = name),
    geom = "point",
    fun.y = "mean",
    show.legend = FALSE,
    size = 3,
    shape = 18
  ) +
  facet_wrap(~name, scales = "free") +
  paper_theme() +
  labs(
    x = "",
    y = "F1 macro score"
  ) +
  guides(color = "none")

ggsave(
  trained_models_scatter / trained_models_swarms ,
  filename = "output/figures/llm_validation/gemini_fine_tune_settings_comb.pdf",
  width = ifelse(export_to_slides, panel_slide_width, default_chart_width + 0.5),
  height = ifelse(export_to_slides, panel_slide_height, default_chart_height * 1.333),
  device = cairo_pdf,
  units = "cm"
)