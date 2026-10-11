# conda activate rstats

library(dplyr)
library(ggplot2)
library(patchwork)
library(kSamples)
library(prettyunits)
library(tidyr)
library(ggh4x)

out_dir <- "C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/"

# Read data
df <- read.csv("C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/global_pixel_sample.csv")

# Remove points in Oceania and Antarctica
df <- subset(df, !continent %in% c("Oceania", "Antarctica"))

# Remove points where median NPV is more than 90
df <- subset(df, p50NPV <= 90)

nrow(df)

# Calculate relative NPV
df$npvProp <- 100 * df$p50NPV / (df$p50PV + df$p50NPV)

# 2D density plots of aridity vs PV an NPV
PV_scatter <- ggplot(df, aes(x = aridity, y = p50PV)) +
	geom_bin2d(show.legend = FALSE, bins = c(400, 100)) +
	scale_fill_distiller(trans = "log10", palette="YlGn", direction = 1) +
    labs(y = "PV (%)") +
	scale_x_continuous(breaks = seq(0, 1, by = 0.5)) +
	coord_cartesian(xlim = c(0, 1)) +
	theme_classic() +
    theme(panel.background = element_blank(), strip.background = element_blank(),
	      axis.text.x = element_blank(), axis.title.x = element_blank(),
		  axis.title.y  = element_text(size=10),
		  plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

NPV_scatter <- ggplot(df, aes(x = aridity, y = p50NPV)) +
	geom_bin2d(show.legend = FALSE, bins = c(400, 90)) +
	scale_fill_distiller(trans = "log10", palette="YlOrBr", direction = 1) +
    labs(y = "NPV (%)") +
	scale_x_continuous(breaks = seq(0, 1, by = 0.5)) +
	coord_cartesian(xlim = c(0, 1), ylim = c(0, 100)) +
	theme_classic() +
    theme(panel.background = element_blank(), strip.background = element_blank(),
	      axis.text.x = element_blank(), axis.title.x = element_blank(),
		  axis.title.y  = element_text(size=10),
		  plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

NPVprop_scatter <- ggplot(df, aes(x = aridity, y = npvProp)) +
	geom_bin2d(show.legend = FALSE, bins = c(400, 100)) +
	scale_fill_distiller(trans = "log10", palette="YlOrBr", direction = 1) +
    labs(x = "Aridity index", y = "rNPV (%)") +
	scale_x_continuous(breaks = seq(0, 1, by = 0.5)) +
	coord_cartesian(xlim = c(0, 1)) +
	theme_classic() +
    theme(panel.background = element_blank(), strip.background = element_blank(),
	      axis.title.y  = element_text(size=10), axis.title.x  = element_text(size=10),
	      plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

scatter_plot <- (PV_scatter / NPV_scatter / NPVprop_scatter) +
  plot_layout(ncol = 1, heights = c(1, 1, 1)) &
  theme(plot.margin = margin(t = 1, r = 1.5, b = 0, l = 0, unit = "pt"))

ggsave(paste0(out_dir, "/PV_NPV_rNPV_vs_aridity.png"), scatter_plot, height = 6.83, width = 3.7, units = "cm", dpi = 600)

# Histograms of PV and NPV in drylands and humidlands
df$climate <- ifelse(df$aridity >= 0.65, "Humid", "Dry")

PV_hist <- ggplot(df, aes(x = p50PV, fill = climate)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 1) +
  scale_fill_manual(values = c("Humid" = "#3498db", "Dry" = "#e74c3c")) +
  labs(x = "PV (%)") +
  coord_cartesian(ylim = c(0, 30000)) +
  theme_classic() +
  theme(axis.title.y = element_blank(), axis.text.y  = element_blank(),
		axis.ticks.y = element_blank(), axis.line.y  = element_blank(),
		legend.title = element_blank(), legend.position = "none",
		axis.title.x  = element_text(size=10),
	    plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

NPV_hist <- ggplot(df, aes(x = p50NPV, fill = climate)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 1) +
  scale_fill_manual(values = c("Humid" = "#3498db", "Dry" = "#e74c3c")) +
  labs(x = "NPV (%)") +
  coord_cartesian(ylim = c(0, 40000)) +
  theme_classic() +
  theme(axis.title.y = element_blank(), axis.text.y  = element_blank(),
		axis.ticks.y = element_blank(), axis.line.y  = element_blank(),
		axis.title.x  = element_text(size=10),
		legend.title = element_blank(), legend.position = "inside",
		legend.position.inside = c(0.8, 0.6), legend.background = element_blank(),
		legend.key.height = unit(0.2, "cm"), legend.key.width = unit(0.3, "cm"),
	    plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

NPVprop_hist <- ggplot(df, aes(x = npvProp, fill = climate)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 1) +
  scale_fill_manual(values = c("Humid" = "#3498db", "Dry" = "#e74c3c")) +
  labs(x = "rNPV (%)") +
  coord_cartesian(ylim = c(0, 30000)) +
  theme_classic() +
  theme(axis.title.y = element_blank(), axis.text.y  = element_blank(),
		axis.ticks.y = element_blank(), axis.line.y  = element_blank(),
		axis.title.x  = element_text(size=10),
		legend.title = element_blank(), legend.position = "none",
	    plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

hist_plot <- (PV_hist / NPV_hist / NPVprop_hist) +
  plot_layout(ncol = 1, heights = c(1, 1, 1)) &
  theme(plot.margin = margin(t = 0, r = 1.5, b = 0, l = 0, unit = "pt"))

ggsave(paste0(out_dir, "/dry_vs_humid_histograms.png"), hist_plot, height = 6.3, width = 3.7, units = "cm", dpi = 600)