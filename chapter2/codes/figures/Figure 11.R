library(arrow)
library(tidyverse)
library(patchwork)
library(binsreg)

source("./codes/functions/chart_themes.R")
source("./codes/functions/diff_in_diff_functions.R")
source("./codes/constants/diff_in_diff_settings.R")

source("./codes/constants/varnames.R")

speech_aggregation <- read_parquet("./data/processed/aggregations/speech_agg.parquet") %>%
  filter(
    year >= 1997,
    year <= 2023
  ) %>%
  mutate(
    date = as.Date(date) # Date datatype is not preserved in parquet it seems
  ) %>%
  arrange(date)

central_bank_year_agg <- read_parquet("./data/processed/aggregations/central_bank_year_agg.parquet") %>%
  filter(
    year >= 1997,
    year <= 2023
  ) %>%
  arrange(central_bank, year)

# Calculate relevant central bank years variables
plot_data <- central_bank_year_agg %>%
  filter(!is.na(inflation)) %>%
  filter(!is.na(ro_cbie_index)) %>%
  mutate(
    advanced = factor(advanced, levels = c(0, 1), labels = c("Emerging and Developing", "Advanced")),
    CBI_discrete = case_when(
      ro_cbie_index > HIGH_CBI ~ "HIGH",
      central_bank == "european central bank" ~ "HIGH",
      ro_cbie_index > 0 ~ "LOW"
    ),
    deviation = inflation - 0.02,
    abs_deviation = abs(deviation),
    deviation_direction = ifelse(deviation > 0, "Upwards", "Downwards"),
  )

fd_bin_reg <- binsreg(financial_dominance, financial_stress, data = as.data.frame(plot_data), by = advanced,
                      cb = c(2, 2), polyreg = 2, , samebinsby = F, bycolors = c(colors[1], colors[2]),
                      nbins = 30, masspoints = "nolocalcheck", plotxrange = c(0, 0.5),
)

fd_plot <- fd_bin_reg$bins_plot +
  paper_theme() +
  ggtitle("B. Financial dominance") +
  scale_color_manual(values = c(colors[2], colors[1])) + # Colors of polynomial fit is wrong order unless set manually. Proably because the package is repsecting the defaults I set.
  paper_theme() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    x = "Financial stress",
    y = "Share",
    group = "",
    color = "",
    fill = "",
  )

fd_plot$layers[[1]]$aes_params$size = 1.2
fd_plot$layers[[4]]$aes_params$size = 1.2

md_bin_reg <- binsreg(monetary_dominance, inflation, data = as.data.frame(plot_data), by = advanced,
                      cb = c(2, 2), polyreg = 2, samebinsby = F, bycolors = c(colors[1], colors[2]),
                      nbins = 15, plotxrange = c(-0.01, 0.15))

md_plot <- md_bin_reg$bins_plot +
  paper_theme() +
  ggtitle("A. Monetary dominance") +
  geom_vline(
    xintercept = 0.02, linetype = "dashed"
  ) +
  annotate("rect", fill = "grey60", alpha = 0.1,
           xmin = 0, xmax = 0.02,
           ymin = -Inf, ymax = Inf) +
  scale_color_manual(values = c(colors[2], colors[1])) + # Colors of polynomial fit is wrong order unless set manually. Proably because the package is repsecting the defaults I set.
  paper_theme() +
  scale_y_continuous(labels = scales::percent) +
  labs(
    x = "HICP inflation rate",
    y = "Share",
    group = "",
    color = "",
    fill = "",
  )

md_plot$layers[[1]]$aes_params$size = 1.2
md_plot$layers[[4]]$aes_params$size = 1.2

(patchwork::wrap_plots(
  md_plot,
  fd_plot,
  guides = "collect"
) & theme(legend.position = 'bottom')) %>%
  ggsave(
    width = panel_chart_width,
    height = default_chart_height * 0.9,
    device = cairo_pdf,
    units = "cm",
    filename = "output/figures/mechanisms/inflation_and_financial_pressures.pdf"
  )