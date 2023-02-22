library(tidyverse)
library(cowplot)
library(reshape2)

fc <- read.csv('C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_models.csv')

# Add dune-swale class from site column
fc$landform <- sapply(strsplit(as.character(fc$site), " "), `[`, 2)

# Calculate RMSE for FC3 
fc3_living_RMSE = sqrt(mean((fc$droneLiving - fc$fc3Living)^2))
fc3_dead_RMSE = sqrt(mean((fc$droneDead - fc$fc3Dead)^2))
fc3_bare_RMSE = sqrt(mean((fc$droneBare - fc$fc3Bare)^2))

# Input RMSE for the Tensorflow model (tfm) using LOOCV
tfm_living_RMSE <- 11.78
tfm_dead_RMSE <- 5.19
tfm_bare_RMSE <- 12.14

# Input RMSE for the Elastic net model (enm) using LOOCV
enm_living_RMSE <- 11.66
enm_dead_RMSE <- 5.39
enm_bare_RMSE <- 11.52

# Plot living, dead and bare drone vs landsat
# - row 1 is fc3 (FC3)
# - row 2 is tensorflow model (TFM)
# - row 2 is the elastic net model (ENM)
fc3_living <- ggplot(fc, aes(x = droneLiving, y = fc3Living)) +
              geom_point(colour = "green") +
              labs(x = "Drone live (%)", y = "FC3 live (%)") +
              theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
              geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
              coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
              geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(fc3_living_RMSE, digits=3), "%"))

fc3_dead <- ggplot(fc, aes(x = droneDead, y = fc3Dead)) +
            geom_point(colour = "blue") +
            labs(x = "Drone dead (%)", y = "FC3 dead (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
            geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(fc3_dead_RMSE, digits=3), "%"))

fc3_bare <- ggplot(fc, aes(x = droneBare, y = fc3Bare)) +
            geom_point(colour = "red") +
            labs(x = "Drone bare (%)", y = "FC3 bare (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
            geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(fc3_bare_RMSE, digits=3), "%"))

tfm_living <- ggplot(fc, aes(x = droneLiving, y = tensorLiving)) +
              geom_point(colour = "green") +
              labs(x = "Drone live (%)", y = "TFM live (%)") +
              theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
              geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
              coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
              geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(tfm_living_RMSE, digits=3), "%"))

tfm_dead <- ggplot(fc, aes(x = droneDead, y = tensorDead)) +
            geom_point(colour = "blue") +
            labs(x = "Drone dead (%)", y = "TFM dead (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
            geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(tfm_dead_RMSE, digits=3), "%"))

tfm_bare <- ggplot(fc, aes(x = droneBare, y = tensorBare)) +
            geom_point(colour = "red") +
            labs(x = "Drone bare (%)", y = "TFM bare (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
            geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(tfm_bare_RMSE, digits=3), "%"))

enm_living <- ggplot(fc, aes(x = droneLiving, y = elasticLiving)) +
              geom_point(colour = "green") +
              labs(x = "Drone live (%)", y = "ENM live (%)") +
              theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
              geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
              coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
              geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(enm_living_RMSE, digits=3), "%"))

enm_dead <- ggplot(fc, aes(x = droneDead, y = elasticDead)) +
            geom_point(colour = "blue") +
            labs(x = "Drone dead (%)", y = "ENM dead (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
            geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(enm_dead_RMSE, digits=3), "%"))

enm_bare <- ggplot(fc, aes(x = droneBare, y = elasticBare)) +
            geom_point(colour = "red") +
            labs(x = "Drone bare (%)", y = "ENM bare (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
            geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(enm_bare_RMSE, digits=3), "%"))

combined_plot <- plot_grid(fc3_living, fc3_dead, fc3_bare,
                           tfm_living, tfm_dead, tfm_bare,
                           enm_living, enm_dead, enm_bare,
                           ncol = 3)
ggsave("C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_drone_comparison.png",
       combined_plot, width = 8, height = 8)

# Combine all the annual data for each site Id

fc2018 <- subset(fc, date == 2018.375)
fc2019 <- subset(fc, date == 2019.375)
fc2020 <- subset(fc, date == 2020.375)
fc2021 <- subset(fc, date == 2021.375)
fc2022 <- subset(fc, date == 2022.375)

fc2018 <- fc2018[order(fc2018$id),]
fc2019 <- fc2019[order(fc2019$id),]
fc2020 <- fc2020[order(fc2020$id),]
fc2021 <- fc2021[order(fc2021$id),]
fc2022 <- fc2022[order(fc2022$id),]

fc2018 <- select(fc2018, id, elasticLiving, elasticDead, elasticBare, landform)
fc2019 <- select(fc2019, id, elasticLiving, elasticDead, elasticBare, landform)
fc2020 <- select(fc2020, id, elasticLiving, elasticDead, elasticBare, landform)
fc2021 <- select(fc2021, id, elasticLiving, elasticDead, elasticBare, landform)
fc2022 <- select(fc2022, id, elasticLiving, elasticDead, elasticBare, landform)

names(fc2018)[2] <- 'Live'
names(fc2018)[3] <- 'Dead'
names(fc2018)[4] <- 'Bare'

names(fc2019)[2] <- 'Live'
names(fc2019)[3] <- 'Dead'
names(fc2019)[4] <- 'Bare'

names(fc2020)[2] <- 'Live'
names(fc2020)[3] <- 'Dead'
names(fc2020)[4] <- 'Bare'

names(fc2021)[2] <- 'Live'
names(fc2021)[3] <- 'Dead'
names(fc2021)[4] <- 'Bare'

names(fc2022)[2] <- 'Live'
names(fc2022)[3] <- 'Dead'
names(fc2022)[4] <- 'Bare'

# Make annual boxplots
                    
fc2018_mod <- melt(fc2018, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2018 <- ggplot(fc2018_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "Landsat ENM cover (%)", title = '2018') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2019_mod <- melt(fc2019, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2019 <- ggplot(fc2019_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2019') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2020_mod <- melt(fc2020, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2020 <- ggplot(fc2020_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2020') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2021_mod <- melt(fc2021, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2021 <- ggplot(fc2021_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2021') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2022_mod <- melt(fc2022, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2022 <- ggplot(fc2022_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2022') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

combined_plot <- plot_grid(box2018, box2019, box2020, box2021, box2022, ncol = 5, align = "h")
ggsave("C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_boxplots.png", combined_plot, width = 8, height = 2.7)

# Make annual boxplots for dune sites only
fc2018_dune <- subset(fc2018, landform == 'DUNE')
fc2019_dune <- subset(fc2019, landform == 'DUNE')
fc2020_dune <- subset(fc2020, landform == 'DUNE')
fc2021_dune <- subset(fc2021, landform == 'DUNE')
fc2022_dune <- subset(fc2022, landform == 'DUNE')

fc2018_mod <- melt(fc2018_dune, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2018 <- ggplot(fc2018_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "Landsat ENM cover (%)", title = '2018') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2019_mod <- melt(fc2019_dune, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2019 <- ggplot(fc2019_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2019') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2020_dmod <- melt(fc2020_dune, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2020 <- ggplot(fc2020_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2020') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2021_mod <- melt(fc2021_dune, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2021 <- ggplot(fc2021_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2021') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2022_mod <- melt(fc2022_dune, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2022 <- ggplot(fc2022_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2022') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

combined_plot <- plot_grid(box2018, box2019, box2020, box2021, box2022, ncol = 5, align = "h")
ggsave("C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_boxplots_dune.png", combined_plot, width = 8, height = 2.7)

# Make annual boxplots for swale sites only
fc2018_swale <- subset(fc2018, landform == 'SWALE')
fc2019_swale <- subset(fc2019, landform == 'SWALE')
fc2020_swale <- subset(fc2020, landform == 'SWALE')
fc2021_swale <- subset(fc2021, landform == 'SWALE')
fc2022_swale <- subset(fc2022, landform == 'SWALE')

fc2018_swale <- melt(fc2018_swale, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2018 <- ggplot(fc2018_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "Landsat ENM cover (%)", title = '2018') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2019_mod <- melt(fc2019_swale, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2019 <- ggplot(fc2019_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2019') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2020_mod <- melt(fc2020_swale, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2020 <- ggplot(fc2020_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2020') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2021_mod <- melt(fc2021_swale, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2021 <- ggplot(fc2021_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2021') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2022_mod <- melt(fc2022_swale, id.vars='id', measure.vars=c('Bare', 'Live', 'Dead'))
box2022 <- ggplot(fc2022_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2022') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

combined_plot <- plot_grid(box2018, box2019, box2020, box2021, box2022, ncol = 5, align = "h")
ggsave("C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_boxplots_swale.png", combined_plot, width = 8, height = 2.7)
