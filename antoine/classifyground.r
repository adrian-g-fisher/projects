# conda create -n lidr -c conda-forge r-lidr r-future r-rgdal r-codetools r-optparse r-dplyr

library(lidR)
library(future)
suppressPackageStartupMessages(library(dplyr))


lowest <- function(res=1) {
  stopifnot(is.numeric(res), length(res) == 1L, res > 0)
  
  f <- function(las) {
    r <- grid_metrics(las, ~.I[which.min(Z)], res=res)
    return(na.omit(r[]))
  }
  
  class(f) <- lidR:::LIDRALGORITHMDEC # Using ::: because not public yet
  return(f)
}


minZ_pmf <- function(las, res=0.1) {
  minZ <- decimate_points(las, lowest(res))
  ws <- seq(3, 12, 3)
  th <- seq(0.05, 1.00, length.out=length(ws))
  minZ <- classify_ground(minZ, algorithm=pmf(ws=ws, th=th))
  joined <- las@data %>% left_join(minZ@data, by=c('X','Y','ReturnNumber'))
  joined[is.na(joined)] <- 0
  las$Classification <- as.integer(joined$Classification.y)
  return(las)
  }

#"S:/fowlers_gap/Sramek_EdwardsExclosures"
lazList <- list.files(path = "C:/Data/edwards_exclosures/salt4_3_exclosure",
                      pattern = "*_group1_densified_point_cloud.las", recursive = TRUE,
					  full.names = TRUE)

for (inLaz in lazList) {
	print(inLaz)
	
	outLaz <- gsub(".las", "_classified.laz", inLaz)
	if (file.exists(outLaz)) {
		print(outLaz)
		} else {
		res <- 0.1
		las <- readLAS(inLaz)
		classified <- minZ_pmf(las, res=res)
		writeLAS(classified, outLaz)
		print(outLaz)
		}
	}