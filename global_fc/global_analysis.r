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

# Remove points outside drylands
df <- subset(df, !dryland %in% c("None"))

# Lists for sorting
dryland_list <- c("Dry subhumid", "Semiarid", "Arid", "Hyperarid")
continent_list <- c("Africa", "Asia", "Australia", "South America", "North America", "Europe")

# Create hex plots of NPV values vs aridity by dryland and continent
df <- transform(df, continent=factor(continent, levels=continent_list))
df <- transform(df, dryland=factor(dryland, levels=dryland_list))

# Rename columns and pivot into longer format
names(df)[names(df) == "p05NPV"] <- "P05"
names(df)[names(df) == "p25NPV"] <- "P25"
names(df)[names(df) == "p50NPV"] <- "P50"
names(df)[names(df) == "p75NPV"] <- "P75"
names(df)[names(df) == "p95NPV"] <- "P95"
dfp <- pivot_longer(df, cols = c(P05, P25, P50, P75, P95),
                    names_to = "Percentiles", values_to = "NPV")

plot1 <- ggplot(dfp, aes(x = Percentiles, y = NPV)) +
	geom_boxplot(show.legend = FALSE) +
	theme_classic() +
    facet_grid(rows = vars(continent), cols = vars(dryland))

ggsave(paste0(out_dir, "/NPV_continent_dryland.png"), plot1, height = 8, width = 8, dpi = 600)