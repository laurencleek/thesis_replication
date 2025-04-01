#begin#Figure 2

library(ggplot2)
library(tidyverse)

source("./codes/functions/chart_themes.R")

# Set parameters
slope_monetary  <- -0.5   # Slope for monetary dominance
slope_financial <-  0.5   # Slope for financial dominance

intercept_fiscal    <- 0.05  # Position for fiscal dominance
intercept_financial <- 0.2   # Intercept for financial dominance
intercept_monetary  <- 0.85  # Increased intercept for monetary dominance (a bit higher)

arrow_pos_x     <- 0.05
font_size_label <- 2.6

# Define small curvature offsets for the quadratic curves
curvature_offset_monetary  <- 0.07   # Slight upward curvature for monetary dominance
curvature_offset_financial <- -0.07  # Slight downward curvature for financial dominance

# Number of points to generate along each curve
n_points <- 100
t_seq <- seq(0, 1, length.out = n_points)

# Endpoints for monetary dominance
P0_monetary <- c(x = 0, y = intercept_monetary)
P2_monetary <- c(x = 1, y = intercept_monetary + slope_monetary)
# The control point is placed at the midpoint in x (0.5) and its y is nudged by a small offset:
control_monetary_y <- (P0_monetary[2] + P2_monetary[2]) / 2 + curvature_offset_monetary

monetary_dominance <- tibble(
  `Independence level` = t_seq,
  `Monetary policy communication` = (1 - t_seq)^2 * P0_monetary[2] +
    2 * (1 - t_seq) * t_seq * control_monetary_y +
    t_seq^2 * P2_monetary[2],
  which = "Monetary dominance"
)

# Endpoints for financial dominance
P0_financial <- c(x = 0, y = intercept_financial)
P2_financial <- c(x = 1, y = intercept_financial + slope_financial)
# The control point is placed at 0.5 in x with a small negative offset
control_financial_y <- (P0_financial[2] + P2_financial[2]) / 2 + curvature_offset_financial

financial_dominance <- tibble(
  `Independence level` = t_seq,
  `Monetary policy communication` = (1 - t_seq)^2 * P0_financial[2] +
    2 * (1 - t_seq) * t_seq * control_financial_y +
    t_seq^2 * P2_financial[2],
  which = "Financial dominance"
)

independence_levels <- seq(0, 1, length.out = 200)
fiscal_dominance <- tibble(
  `Monetary policy communication` = intercept_fiscal + 0.015 * sin(60 * pi * independence_levels),  # Smaller amplitude and tighter period
  `Independence level` = independence_levels,
  which = "Fiscal dominance"
)

# Combine all data
dominance_data <- bind_rows(
  monetary_dominance,
  fiscal_dominance,
  financial_dominance
)

# For the arrow annotations we need the y-values at t = 1 - arrow_pos_x.
t_arrow <- 1 - arrow_pos_x
monetary_arrow_y <- (1 - t_arrow)^2 * P0_monetary[2] + 2 * (1 - t_arrow) * t_arrow * control_monetary_y + t_arrow^2 * P2_monetary[2]
financial_arrow_y <- (1 - t_arrow)^2 * P0_financial[2] + 2 * (1 - t_arrow) * t_arrow * control_financial_y + t_arrow^2 * P2_financial[2]

# Plotting
ggplot(dominance_data, aes(x = `Independence level`, y = `Monetary policy communication`, color = which)) +
  geom_line(data = filter(dominance_data, which == "Monetary dominance"), show.legend = FALSE) +
  geom_line(data = filter(dominance_data, which == "Financial dominance"), show.legend = FALSE) +
  # To include the fiscal dominance curve, uncomment the next line:
  # geom_line(data = filter(dominance_data, which == "Fiscal dominance"), alpha = 0.5, show.legend = FALSE) +
  paper_theme() +
  scale_x_continuous(breaks = c(0.06, 0.94), labels = c("low", "high")) +
  scale_y_continuous(breaks = c(0.03, 0.97), labels = c("low", "high")) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.ticks = element_blank(),
    plot.margin = unit(c(0.1, 3.1, 0.1, 0.1), "cm"),
    axis.title = element_text(face = "bold", family = "XCharter Math", colour = "black")
  ) +
  annotate(geom = "text",
           family = "XCharter Math",
           x = 1.07,
           y = P2_monetary[2] + 0.02,  # slightly above the end of the monetary curve
           label = "Monetary dominance",
           size = font_size_label,
           hjust = 0) +
  annotate(
    geom = "curve",
    x = 1.06,
    y = P2_monetary[2] + 0.02,
    xend = 1 - arrow_pos_x,
    yend = monetary_arrow_y,
    curvature = 0.3,
    linewidth = 0.3,
    arrow = arrow(length = unit(2, "mm"))
  ) +
  annotate(geom = "text",
           family = "XCharter Math",
           x = 1.07,
           y = P2_financial[2] - 0.05,  # slightly below the end of the financial curve
           label = "Financial dominance",
           size = font_size_label,
           hjust = 0) +
  annotate(
    geom = "curve",
    x = 1.06,
    y = P2_financial[2] - 0.05,
    xend = 1 - arrow_pos_x,
    yend = financial_arrow_y,
    curvature = -0.3,
    linewidth = 0.3,
    arrow = arrow(length = unit(2, "mm"))
  ) +
  # annotate(geom = "text",
  #          family = "XCharter Math",
  #          x = 1.07,
  #          y = intercept_fiscal - 0.03,
  #          label = "Fiscal dominance",
  #          size = font_size_label,
  #          hjust = 0) +
  # annotate(
  #   geom = "curve",
  #   x = 1.06,
  #   y = intercept_fiscal - 0.03,
  #   xend = 1 - arrow_pos_x,
  #   yend = intercept_fiscal,
  #   curvature = -0.3,
  #   linewidth = 0.3,
  #   arrow = arrow(length = unit(2, "mm"))
  # ) +
  coord_cartesian(xlim = c(0, 1),  # focus the x-axis on the range of interest
                  ylim = c(0, 1),
                  clip = 'off') +
  scale_color_manual(values = c(colors[2], colors[1]))

ggsave(
  filename = "output/figures/mechanisms/dominance_scheme.pdf",
  width = default_chart_width * 0.95,
  height = default_chart_height * 0.8,
  device = cairo_pdf,
  units = "cm"
)