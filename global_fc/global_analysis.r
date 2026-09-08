# conda activate rstats

library(dplyr)
library(ggplot2)
library(patchwork)
library(kSamples)
library(prettyunits)
library(tidyr)
library(ggh4x)

out_dir <- "C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/"

# Read data and fix vallues over 100
df <- read.csv("C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/global_pixel_sample.csv")
df$p05PV[df$p05PV > 100] <- 100
df$p05NPV[df$p05NPV > 100] <- 100
df$p05BS[df$p05BS > 100] <- 100
df$p25PV[df$p25PV > 100] <- 100
df$p25NPV[df$p25NPV > 100] <- 100
df$p25BS[df$p25BS > 100] <- 100
df$p50PV[df$p50PV > 100] <- 100
df$p50NPV[df$p50NPV > 100] <- 100
df$p50BS[df$p50BS > 100] <- 100
df$p75PV[df$p75PV > 100] <- 100
df$p75NPV[df$p75NPV > 100] <- 100
df$p75BS[df$p75BS > 100] <- 100
df$p95PV[df$p95PV > 100] <- 100
df$p95NPV[df$p95PV > 100] <- 100
df$p95BS[df$p95BS > 100] <- 100

# Remove points in Oceania and Antarctica
df <- subset(df, !continent %in% c("Oceania", "Antarctica"))
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