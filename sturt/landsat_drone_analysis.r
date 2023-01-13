library(tidyverse)
library(cowplot)
library(reshape2)

fc <- read.csv('C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/comparison_fc_v3.csv')
fc$satTotal <- fc$satPV + fc$satNPV
fc$droneTotal <- fc$droneAlive + fc$droneDead

# Add dune-swale class from Ident column
fc$landform <- sapply(strsplit(as.character(fc$Ident), " "), `[`, 2)

# Develop model for PV
PVmodel <- lm(data = fc, satPV ~ 0 + droneAlive)
summary(PVmodel)
fc$predicted_PV <- predict(PVmodel)

# Develop model for total cover
totalModel <- lm(data = fc, satTotal ~ droneTotal)
summary(totalModel)
fc$predicted_total <- predict(totalModel)

# Calculate predicted NPV from the other models
fc$predicted_NPV <- fc$predicted_total - fc$predicted_PV

# Calculate RMSE using LOOCV
fc$loocv_PV <- 0
fc$loocv_NPV <- 0
fc$loocv_total <- 0
for (i in 1:nrow(fc)) {
    training <- fc[-i,]
    p <- lm(data = training, satPV ~ 0 + droneAlive)
    fc[i,]$loocv_PV <- predict(p, newdata=fc[i,])
    t <- lm(data = training, satTotal ~ droneTotal)
    fc[i,]$loocv_total <- predict(t, newdata=fc[i,])
    fc[i,]$loocv_NPV <- fc[i,]$loocv_total - fc[i,]$loocv_PV
    }

PV_RMSE = sqrt(mean((fc$satPV - fc$loocv_PV)^2))
NPV_RMSE = sqrt(mean((fc$satNPV - fc$loocv_NPV)^2))
total_RMSE = sqrt(mean((fc$satTotal - fc$loocv_total)^2))

# Plot PV, NPV and total cover with RMSE
PV_plot <- ggplot(fc, aes(x = predicted_PV, y = satPV)) +
           geom_point(colour = "grey") +
           labs(x = "Drone PV (%)", y = "Landsat PV (%)") +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
           geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
           coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
           geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(PV_RMSE, digits=2), "%"))

NPV_plot <- ggplot(fc, aes(x = predicted_NPV, y = satNPV)) +
            geom_point(colour = "grey") +
            labs(x = "Drone NPV (%)", y = "Landsat NPV (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
            geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(NPV_RMSE, digits=2), "%"))

total_plot <- ggplot(fc, aes(x = predicted_total, y = satTotal)) +
              geom_point(colour = "grey") +
              labs(x = "Drone total cover (%)", y = "Landsat total cover (%)") +
              theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
              geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
              coord_cartesian(xlim=c(0, 100), ylim=c(0, 100)) +
              geom_text(x=2, y=97, hjust = 0, label=paste("RMSE =", format(total_RMSE, digits=2), "%"))

combined_plot <- plot_grid(PV_plot, NPV_plot, total_plot, ncol = 3, align = "h")
ggsave("C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_drone_comparison.png", combined_plot, width = 8, height = 2.7)

# Combine all the annual data for each site Id

fc2018 <- subset(fc, date == 2018.375)
fc2019 <- subset(fc, date == 2019.375)
fc2020 <- subset(fc, date == 2020.375)
fc2021 <- subset(fc, date == 2021.375)
fc2022 <- subset(fc, date == 2022.375)

fc2018 <- fc2018[order(fc2018$Id),]
fc2019 <- fc2019[order(fc2019$Id),]
fc2020 <- fc2020[order(fc2020$Id),]
fc2021 <- fc2021[order(fc2021$Id),]
fc2022 <- fc2022[order(fc2022$Id),]

fc2018 <- select(fc2018, Id, satPV, satNPV, satBare, satTotal, predicted_PV, predicted_total, predicted_NPV, landform)
fc2019 <- select(fc2019, Id, satPV, satNPV, satBare, satTotal, predicted_PV, predicted_total, predicted_NPV, landform)
fc2020 <- select(fc2020, Id, satPV, satNPV, satBare, satTotal, predicted_PV, predicted_total, predicted_NPV, landform)
fc2021 <- select(fc2021, Id, satPV, satNPV, satBare, satTotal, predicted_PV, predicted_total, predicted_NPV, landform)
fc2022 <- select(fc2022, Id, satPV, satNPV, satBare, satTotal, predicted_PV, predicted_total, predicted_NPV, landform)

names(fc2018)[2] <- 'satPV_2018'
names(fc2018)[3] <- 'satNPV_2018'
names(fc2018)[4] <- 'satBare_2018'
names(fc2018)[5] <- 'satTotal_2018'
names(fc2018)[6] <- 'dronePV_2018'
names(fc2018)[7] <- 'droneNPV_2018'
names(fc2018)[8] <- 'droneTotal_2018'

names(fc2019)[2] <- 'satPV_2019'
names(fc2019)[3] <- 'satNPV_2019'
names(fc2019)[4] <- 'satBare_2019'
names(fc2019)[5] <- 'satTotal_2019'
names(fc2019)[6] <- 'dronePV_2019'
names(fc2019)[7] <- 'droneNPV_2019'
names(fc2019)[8] <- 'droneTotal_2019'

names(fc2020)[2] <- 'satPV_2020'
names(fc2020)[3] <- 'satNPV_2020'
names(fc2020)[4] <- 'satBare_2020'
names(fc2020)[5] <- 'satTotal_2020'
names(fc2020)[6] <- 'dronePV_2020'
names(fc2020)[7] <- 'droneNPV_2020'
names(fc2020)[8] <- 'droneTotal_2020'

names(fc2021)[2] <- 'satPV_2021'
names(fc2021)[3] <- 'satNPV_2021'
names(fc2021)[4] <- 'satBare_2021'
names(fc2021)[5] <- 'satTotal_2021'
names(fc2021)[6] <- 'dronePV_2021'
names(fc2021)[7] <- 'droneNPV_2021'
names(fc2021)[8] <- 'droneTotal_2021'

names(fc2022)[2] <- 'satPV_2022'
names(fc2022)[3] <- 'satNPV_2022'
names(fc2022)[4] <- 'satBare_2022'
names(fc2022)[5] <- 'satTotal_2022'
names(fc2022)[6] <- 'dronePV_2022'
names(fc2022)[7] <- 'droneNPV_2022'
names(fc2022)[8] <- 'droneTotal_2022'

# Calculate annual changes in drone and landsat PV and NPV

fc_change <- Reduce(function(x, y) merge(x, y, all=TRUE), list(fc2018, fc2019, fc2020, fc2021, fc2022))
                    
fc_change$satPV_2018_2019 <- fc_change$satPV_2019 - fc_change$satPV_2018
fc_change$satPV_2019_2020 <- fc_change$satPV_2020 - fc_change$satPV_2019
fc_change$satPV_2020_2021 <- fc_change$satPV_2021 - fc_change$satPV_2020
fc_change$satPV_2021_2022 <- fc_change$satPV_2022 - fc_change$satPV_2021

fc_change$dronePV_2018_2019 <- fc_change$dronePV_2019 - fc_change$dronePV_2018
fc_change$dronePV_2019_2020 <- fc_change$dronePV_2020 - fc_change$dronePV_2019
fc_change$dronePV_2020_2021 <- fc_change$dronePV_2021 - fc_change$dronePV_2020
fc_change$dronePV_2021_2022 <- fc_change$dronePV_2022 - fc_change$dronePV_2021

fc_change$satNPV_2018_2019 <- fc_change$satNPV_2019 - fc_change$satNPV_2018
fc_change$satNPV_2019_2020 <- fc_change$satNPV_2020 - fc_change$satNPV_2019
fc_change$satNPV_2020_2021 <- fc_change$satNPV_2021 - fc_change$satNPV_2020
fc_change$satNPV_2021_2022 <- fc_change$satNPV_2022 - fc_change$satNPV_2021

fc_change$droneNPV_2018_2019 <- fc_change$droneNPV_2019 - fc_change$droneNPV_2018
fc_change$droneNPV_2019_2020 <- fc_change$droneNPV_2020 - fc_change$droneNPV_2019
fc_change$droneNPV_2020_2021 <- fc_change$droneNPV_2021 - fc_change$droneNPV_2020
fc_change$droneNPV_2021_2022 <- fc_change$droneNPV_2022 - fc_change$droneNPV_2021

# Combine all change data into columns

change_2018_2019 <- select(fc_change, Id, satPV_2018_2019, dronePV_2018_2019, satNPV_2018_2019, droneNPV_2018_2019)
change_2019_2020 <- select(fc_change, Id, satPV_2019_2020, dronePV_2019_2020, satNPV_2019_2020, droneNPV_2019_2020)
change_2020_2021 <- select(fc_change, Id, satPV_2020_2021, dronePV_2020_2021, satNPV_2020_2021, droneNPV_2020_2021)
change_2021_2022 <- select(fc_change, Id, satPV_2021_2022, dronePV_2021_2022, satNPV_2021_2022, droneNPV_2021_2022)

change_2018_2019$years <- '2018-2019'
change_2019_2020$years <- '2019-2020'
change_2020_2021$years <- '2020-2021'
change_2021_2022$years <- '2021-2022'

names(change_2018_2019)[2] <- 'satPV_change'
names(change_2018_2019)[3] <- 'dronePV_change'
names(change_2018_2019)[4] <- 'satNPV_change'
names(change_2018_2019)[5] <- 'droneNPV_change'

names(change_2019_2020)[2] <- 'satPV_change'
names(change_2019_2020)[3] <- 'dronePV_change'
names(change_2019_2020)[4] <- 'satNPV_change'
names(change_2019_2020)[5] <- 'droneNPV_change'

names(change_2020_2021)[2] <- 'satPV_change'
names(change_2020_2021)[3] <- 'dronePV_change'
names(change_2020_2021)[4] <- 'satNPV_change'
names(change_2020_2021)[5] <- 'droneNPV_change'

names(change_2021_2022)[2] <- 'satPV_change'
names(change_2021_2022)[3] <- 'dronePV_change'
names(change_2021_2022)[4] <- 'satNPV_change'
names(change_2021_2022)[5] <- 'droneNPV_change'

change <- rbind(change_2018_2019, change_2019_2020, change_2020_2021, change_2021_2022)
change <- na.omit(change)

change$satTotal_change <- change$satPV_change + change$satNPV_change
change$droneTotal_change <- change$dronePV_change + change$droneNPV_change
                    
# Calculate RMSE and plot
PV_change_RMSE = sqrt(mean((change$satPV_change - change$dronePV_change)^2))
NPV_change_RMSE = sqrt(mean((change$satNPV_change - change$droneNPV_change)^2))
total_change_RMSE = sqrt(mean((change$satTotal_change - change$droneTotal_change)^2))

PV_plot <- ggplot(change, aes(x = satPV_change, y = dronePV_change)) +
           geom_point(colour = "grey") +
           labs(x = "Change in drone PV (%)", y = "Change in Landsat PV (%)") +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
           geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
           coord_cartesian(xlim=c(-15, 30), ylim=c(-15, 30)) +
           geom_text(x=-15, y=29, hjust = 0, label=paste("RMSE =", format(PV_change_RMSE, digits=2), "%"))

NPV_plot <- ggplot(change, aes(x = satNPV_change, y = droneNPV_change)) +
            geom_point(colour = "grey") +
            labs(x = "Change in drone NPV (%)", y = "Change in Landsat NPV (%)") +
            theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
            geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
            coord_cartesian(xlim=c(-20, 40), ylim=c(-20, 40)) +
            geom_text(x=-20, y=39, hjust = 0, label=paste("RMSE =", format(NPV_change_RMSE, digits=2), "%"))

total_plot <- ggplot(change, aes(x = satTotal_change, y = droneTotal_change)) +
              geom_point(colour = "grey") +
              labs(x = "Change in drone total (%)", y = "Change in Landsat total (%)") +
              theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
              geom_abline(slope = 1, intercept = 0, color = 'black', size = 0.3) +
              coord_cartesian(xlim=c(-20, 40), ylim=c(-20, 40)) +
              geom_text(x=-20, y=39, hjust = 0, label=paste("RMSE =", format(total_change_RMSE, digits=2), "%"))

combined_plot <- plot_grid(PV_plot, NPV_plot, total_plot, ncol = 3, align = "h")
ggsave("C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_drone_change_comparison.png", combined_plot, width = 8, height = 2.7)
     
# Make annual boxplots
                    
names(fc2018)[2] <- 'PV'
names(fc2018)[3] <- 'NPV'
names(fc2018)[4] <- 'Bare'
fc2018_mod <- melt(fc2018, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2018 <- ggplot(fc2018_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "Landsat cover (%)", title = '2018') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2019$Bare <- 100 - (fc2019$satTotal_2019)
names(fc2019)[2] <- 'PV'
names(fc2019)[3] <- 'NPV'
fc2019_mod <- melt(fc2019, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2019 <- ggplot(fc2019_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2019') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2020$Bare <- 100 - (fc2020$satTotal_2020)
names(fc2020)[2] <- 'PV'
names(fc2020)[3] <- 'NPV'
fc2020_mod <- melt(fc2020, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2020 <- ggplot(fc2020_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2020') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2021$Bare <- 100 - (fc2021$satTotal_2021)
names(fc2021)[2] <- 'PV'
names(fc2021)[3] <- 'NPV'
fc2021_mod <- melt(fc2021, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2021 <- ggplot(fc2021_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2021') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2022$Bare <- 100 - (fc2022$satTotal_2022)
names(fc2022)[2] <- 'PV'
names(fc2022)[3] <- 'NPV'
fc2022_mod <- melt(fc2022, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
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

fc2018_mod <- melt(fc2018_dune, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2018 <- ggplot(fc2018_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "Landsat cover (%)", title = '2018') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2019_mod <- melt(fc2019_dune, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2019 <- ggplot(fc2019_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2019') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2020_mod <- melt(fc2020_dune, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2020 <- ggplot(fc2020_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2020') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2021_mod <- melt(fc2021_dune, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2021 <- ggplot(fc2021_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2021') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2022_mod <- melt(fc2022_dune, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
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

fc2018_mod <- melt(fc2018_swale, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2018 <- ggplot(fc2018_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "Landsat cover (%)", title = '2018') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2019_mod <- melt(fc2019_swale, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2019 <- ggplot(fc2019_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2019') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2020_mod <- melt(fc2020_swale, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2020 <- ggplot(fc2020_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2020') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2021_mod <- melt(fc2021_swale, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2021 <- ggplot(fc2021_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2021') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

fc2022_mod <- melt(fc2022_swale, id.vars='Id', measure.vars=c('Bare', 'PV', 'NPV'))
box2022 <- ggplot(fc2022_mod) +
           geom_boxplot(aes(x=variable, y=value, color=variable)) +
           labs(x = "", y = "", title = '2022') +
           coord_cartesian(ylim=c(0, 100)) +
           theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position="none")

combined_plot <- plot_grid(box2018, box2019, box2020, box2021, box2022, ncol = 5, align = "h")
ggsave("C:/Users/Adrian/OneDrive - UNSW/Documents/papers/preparation/wild_deserts_vegetation_change/landsat_boxplots_swale.png", combined_plot, width = 8, height = 2.7)

