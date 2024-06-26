library(dplyr)
library(ggplot2)

csvfile <- 'C:\\Users\\Adrian\\OneDrive - UNSW\\Documents\\witchelina\\awp_grazing_pressure\\redo_analyses_epsg3577\\awp_analysis_epsg3577.csv'
awp_data <- read.csv(csvfile)

model <- lm(data=awp_data, NPV_mean_before_grazing ~ Distance + Paddock)
summary(model)

model <- lm(data=awp_data, NPV_mean_after_grazing ~ Distance + Paddock)
summary(model)

newx = seq(min(awp_data$Distance), max(awp_data$Distance), by = 30)
conf_interval <- predict(model, newdata=data.frame(x=newx), interval="confidence", level = 0.95)



plot(awp_data$Distance, y, xlab="x", ylab="y", main="Regression")
abline(model, col="lightblue")
lines(newx, conf_interval[,2], col="blue", lty=2)
lines(newx, conf_interval[,3], col="blue", lty=2)
