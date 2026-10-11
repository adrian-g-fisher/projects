# conda activate rstats

library(dplyr)
library(ggplot2)
library(patchwork)
library(kSamples)
library(prettyunits)
library(tidyr)
library(ggh4x)
library(lubridate)

out_dir <- "C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/"

g_df <- read.csv("C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/gobabeb_fc_timeseries.csv")
g_df <- setNames(g_df, c("date_text", "PV", "NPV", "BS", "TV", "Rainfall"))
g_df$date_text <- substr(g_df$date_text, start = 1, stop = 10)
g_df$Date <- ymd(g_df$date_text)
g_df_monthly <- g_df %>%
  mutate(Date = floor_date(Date, "month")) %>%
  group_by(Date) %>%
  summarise(Rainfall = sum(Rainfall, na.rm = TRUE),
            PV = sum(PV, na.rm = TRUE),
			NPV = sum(NPV, na.rm = TRUE))
g_df_monthly$Location <- "Gobabeb"
g_df_monthly$NPV[g_df_monthly$NPV == 0] <- NA

j_df <- read.csv("C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/jornada_fc_timeseries.csv")
j_df <- setNames(j_df, c("date_text", "PV", "NPV", "BS", "TV", "Rainfall"))
j_df$date_text <- substr(j_df$date_text, start = 1, stop = 10)
j_df$Date <- ymd(j_df$date_text)
j_df_monthly <- j_df %>%
  mutate(Date = floor_date(Date, "month")) %>%
  group_by(Date) %>%
  summarise(Rainfall = sum(Rainfall, na.rm = TRUE),
            PV = sum(PV, na.rm = TRUE),
			NPV = sum(NPV, na.rm = TRUE))
j_df_monthly$Location <- "Jornada"
j_df_monthly$NPV[j_df_monthly$NPV == 0] <- NA

f_df <- read.csv("C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/fowlers_fc_timeseries.csv")
f_df <- setNames(f_df, c("date_text", "PV", "NPV", "BS", "TV", "Rainfall"))
f_df$date_text <- substr(f_df$date_text, start = 1, stop = 10)
f_df$Date <- ymd(f_df$date_text)
f_df_monthly <- f_df %>%
  mutate(Date = floor_date(Date, "month")) %>%
  group_by(Date) %>%
  summarise(Rainfall = sum(Rainfall, na.rm = TRUE),
            PV = sum(PV, na.rm = TRUE),
			NPV = sum(NPV, na.rm = TRUE))
f_df_monthly$Location <- "Jornada"
f_df_monthly$NPV[f_df_monthly$NPV == 0] <- NA

median(j_df_monthly$PV, na.rm = TRUE) 
median(j_df_monthly$NPV, na.rm = TRUE)
max(j_df_monthly$PV, na.rm = TRUE) 
max(j_df_monthly$NPV, na.rm = TRUE)

scale_factor = 0.3
g_timeseries <- ggplot(g_df_monthly, aes(x = Date)) +
	geom_col(aes(y = Rainfall * scale_factor), fill = "deepskyblue") +    
	geom_line(aes(y = PV, group = 1), linetype = 1, color='darkgreen') +
	geom_line(aes(y = NPV, group = 1), linetype = 1, color='darkgoldenrod') +
	scale_y_continuous(name = "",
	                   sec.axis = sec_axis(~ . / scale_factor, name = "                                    Monthly rainfall (mm)")) +
	geom_text(label = "Gobabeb", y = 19.5, x = as.Date("2009-05-01"), hjust = 0, size=10/.pt) +
    coord_cartesian(xlim = c(as.Date("2010-01-01"), as.Date("2026-01-01"))) +
	theme_classic() +
    theme(panel.background = element_blank(),
	      strip.background = element_blank(), strip.text = element_blank(),
	      axis.text = element_text(size=10), axis.title  = element_text(size=10),
	      axis.text.x = element_blank(), axis.title.x = element_blank(),
		  plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

f_timeseries <- ggplot(f_df_monthly, aes(x = Date)) +
	geom_col(aes(y = Rainfall * scale_factor), fill = "deepskyblue") +    
	geom_line(aes(y = PV, group = 1), linetype = 1, color='darkgreen') +
	geom_line(aes(y = NPV, group = 1), linetype = 1, color='darkgoldenrod') +
	scale_y_continuous(name = "",
	                   sec.axis = sec_axis(~ . / scale_factor, name = "")) +
	geom_text(label = "Fowlers Gap", y = 60, x = as.Date("2009-05-01"), hjust = 0, size=10/.pt) +
    coord_cartesian(xlim = c(as.Date("2010-01-01"), as.Date("2026-01-01"))) +
	theme_classic() +
    theme(panel.background = element_blank(),
	      strip.background = element_blank(), strip.text = element_blank(),
	      axis.text = element_text(size=10), axis.title  = element_text(size=10),
	      axis.text.x = element_blank(), axis.title.x = element_blank(),
		  plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

j_timeseries <- ggplot(j_df_monthly, aes(x = Date)) +
	geom_col(aes(y = Rainfall * scale_factor), fill = "deepskyblue") +    
	geom_line(aes(y = PV, group = 1), linetype = 1, color='darkgreen') +
	geom_line(aes(y = NPV, group = 1), linetype = 1, color='darkgoldenrod') +
	scale_y_continuous(name = "                                        Vegetation cover (%)",
	                   sec.axis = sec_axis(~ . / scale_factor, name = "")) +
	geom_text(label = "Jornada", y = 51, x = as.Date("2009-05-01"), hjust = 0, size=10/.pt) +
	coord_cartesian(xlim = c(as.Date("2010-01-01"), as.Date("2026-01-01"))) +
	theme_classic() +
    theme(panel.background = element_blank(), axis.title.x = element_blank(),
	      strip.background = element_blank(), strip.text = element_blank(),
	      axis.text = element_text(size=10), axis.title  = element_text(size=10),
		  plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

timeseries <- (g_timeseries / f_timeseries / j_timeseries) +
  plot_layout(ncol = 1, heights = c(1, 1, 1)) &
  theme(plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))

ggsave(paste0(out_dir, "example_timeseries.png"), timeseries, height = 6, width = 15.8, units = "cm", dpi = 600)