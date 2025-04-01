default_chart_width <-  12 # In cm
default_chart_height <-  9 # in cm
panel_chart_height <- 22 #cm
panel_chart_width  <- 15 #cm

panel_slide_width <- 18
panel_slide_height <- 13

# Slides!
export_to_slides <- FALSE # If set to true plot settings will change
patchwork2x1_design <- "AABB\n#CC#"
dominance_to_plot <-  c("monetary_dominance", "financial_dominance")

panel_chart_height = panel_chart_height * (1/3 * length(dominance_to_plot))

HIST_FILL_ALPHA <- 0.7


paper_theme <- function() {
   theme_minimal() %+replace%
     theme(
       plot.margin= margin(0,0,0,0, "pt"),
       legend.margin =  margin(0, 0, 0, 0, "pt"),
       text = element_text(size=rel(1), family="XCharter Math", color = "black"),
       axis.text = element_text(colour="black"),
       plot.title = element_text(size=rel(0.9), hjust = 0, margin = margin(t = 0, r = 0, b = 10, l = 0)),
       plot.title.position = "plot",
       axis.title = element_text(size = rel(0.75), color = "black"),
       #title = element_text(face = "bold"),
       #legend.title = element_text(face="bold"),
       legend.text = element_text(size = rel(0.7)),
       axis.text.x = element_text(size = rel(0.75), margin = margin(t = 1, r = 0, b = 0, l = 0)),
       axis.text.y = element_text(size = rel(0.75), margin = margin(t = 0, r = 1, b = 0, l = 0), hjust = 1),
       axis.line = element_line(colour = "black", linewidth = 0.25),
       axis.ticks = element_line(colour = "black", linewidth = 0.25),
       panel.grid.major = element_line(colour = "grey75", linetype = "dotted", linewidth =0.25),
       panel.grid.major.x = element_blank(),
       panel.grid.minor = element_blank(),
       axis.ticks.length=unit(.1, "cm"),
       panel.border = element_blank(),
       legend.position = "top",
       legend.spacing.y = unit(0, 'cm'),
       legend.key.size = unit(0.75,"line")
     )
}

paper_theme_map <- function() {
   theme_minimal() %+replace%
     theme(
       text = element_text(size=rel(1), family="XCharter Math", color = "black"),
       axis.text = element_text(colour="black", size=rel(0.7)),
       plot.title = element_text(size=rel(0.9), hjust = 0, margin = margin(t = 0, r = 0, b = 5, l = 0)),
       plot.title.position = "plot",
       legend.position = "bottom",
       #text = element_text(size=rel(1), family="LM Roman 10", color = "black"),
       legend.margin=margin(0,0,0,0),
       legend.box.margin=margin(-15,-15,-15,-15),
       plot.margin = margin(-25,-15,-15,-15),
       legend.text = element_text(size = rel(0.7)),
       axis.title = element_text(size = rel(0.75), color = "black"),
       legend.spacing.x = unit(0.0, units = "cm")
     )
}

# Set font
#if (.Platform$OS.type == "windows") {
#  windowsFonts("Computer Modern" = windowsFont("NewComputerModern10"))
#}

colors <- c(
  "#156082",
  "#804545",
  "#458064",
  "#EAB676",
  "#6d4991",
  "#F27724",
  "#5BAAD9",
  "#A288C1",
  "#951B22",
  "#3E4D4F")

options(ggplot2.discrete.colour= colors)
options(ggplot2.discrete.fill= colors)
