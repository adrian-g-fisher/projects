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

# Compare median NPV distributions in drylands and humidlands
df$climate <- ifelse(df$dryland == "None", "Humid", "Drylands")

plot1 <- ggplot(df, aes(x = p50NPV, fill = climate)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 1, color = "white") +
  scale_fill_manual(values = c("Humid" = "#3498db", "Drylands" = "#e74c3c")) +
  labs(x = "Median NPV (%)", y = "") +
  coord_cartesian(ylim = c(0, 30000)) +
  theme_classic() +
  theme(axis.title.y = element_blank(), axis.text.y  = element_blank(),
		axis.ticks.y = element_blank(), axis.line.y  = element_blank(),
		legend.title = element_blank(), legend.position = "inside",
        legend.position.inside = c(0.85, 0.85))
ggsave(paste0(out_dir, "/dry_vs_humid_NPV.png"), plot1, height = 1.5, width = 4, dpi = 600)

plot1 <- ggplot(df, aes(x = p50PV, fill = climate)) +
  geom_histogram(position = "identity", alpha = 0.5, binwidth = 1, color = "white") +
  scale_fill_manual(values = c("Humid" = "#3498db", "Drylands" = "#e74c3c")) +
  labs(x = "Median PV (%)", y = "") +
  coord_cartesian(ylim = c(0, 30000)) +
  theme_classic() +
  theme(axis.title.y = element_blank(), axis.text.y  = element_blank(),
		axis.ticks.y = element_blank(), axis.line.y  = element_blank(),
		legend.title = element_blank(), legend.position = "inside",
        legend.position.inside = c(0.85, 0.85))
ggsave(paste0(out_dir, "/dry_vs_humid_PV.png"), plot1, height = 1.5, width = 4, dpi = 600)




# # Remove points outside drylands
# df <- subset(df, !dryland %in% c("None"))

# # Lists for sorting
# dryland_list <- c("Dry subhumid", "Semiarid", "Arid", "Hyperarid")
# continent_list <- c("Africa", "Asia", "Australia", "South America", "North America", "Europe")

# # Boxplots of NPV values vs aridity by dryland and continent
# df <- transform(df, continent=factor(continent, levels=continent_list))
# df <- transform(df, dryland=factor(dryland, levels=dryland_list))

# names(df)[names(df) == "p05NPV"] <- "P05"
# names(df)[names(df) == "p25NPV"] <- "P25"
# names(df)[names(df) == "p50NPV"] <- "P50"
# names(df)[names(df) == "p75NPV"] <- "P75"
# names(df)[names(df) == "p95NPV"] <- "P95"

# dfp <- pivot_longer(df, cols = c(P05, P25, P50, P75, P95),
                    # names_to = "Percentiles", values_to = "NPV")

# plot2 <- ggplot(dfp, aes(x = Percentiles, y = NPV)) +
	# geom_boxplot(show.legend = FALSE) +
	# theme_classic() +
    # facet_grid(rows = vars(continent), cols = vars(dryland))

# ggsave(paste0(out_dir, "/NPV_continent_dryland.png"), plot2, height = 8, width = 8, dpi = 600)