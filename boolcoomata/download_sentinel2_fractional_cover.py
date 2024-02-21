#!/usr/bin/env python

import os
import sys
from osgeo import gdal, ogr
from datetime import datetime

# Inputs and outputs
polygon = r'S:\boolcoomatta\shapefiles\boolcoomatta_aoi_albers.shp'
dstDir = r'S:\boolcoomatta\sentinel2_seasonal_fractional_cover'

# Read in shapefile and get bounding box
basename = os.path.basename(polygon).replace(r'.shp', '')
ds = ogr.Open(polygon)
for lyr in ds:
    (xmin, xmax, ymin, ymax) = lyr.GetExtent()
bbox = [int(round(xmin)), int(round(ymax)), int(round(xmax)), int(round(ymin))]
ds = None

# Construct dateList for all seasonal dates
start = 201603201605
end = 202309202311
dateList = []
for y1 in range(1987, 2024):
    for m1 in range(3, 13, 3):
        if m1 < 12:
            y2 = y1
            m2 = m1 + 2
        else:
            y2 = y1 + 1
            m2 = 2
        date = int(r'%i%02d%i%02d'%(y1, m1, y2, m2))
        if date >= start and date <= end:
            dateList.append(date)

# For each date make the image subset
srcDir = r'/vsicurl/https://data.tern.org.au/rs/public/data/sentinel2/seasonal_fractional_cover/fractional_cover/nsw/'

for date in dateList:
    srcImage = r'cvmsre_nsw_m%i_acaa2.tif'%date
    srcFile = os.path.join(srcDir, srcImage)
    dstFile = os.path.join(dstDir, srcImage.replace(r'.tif', r'_subset.tif'))
    
    if os.path.exists(dstFile) is False:
        src_ds = gdal.Open(srcFile)
        dst_ds = gdal.Translate(dstFile, src_ds, projWin=bbox)
        dst_ds = None
        src_ds = None 

print('Image subsets downloaded')