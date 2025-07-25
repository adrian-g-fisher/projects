#!/usr/bin/env python

import os
import sys
from osgeo import gdal, ogr
from datetime import datetime

# Inputs and outputs
polygon = r'C:\Users\Adrian\OneDrive - UNSW\Documents\publications\published\2023_01_wild_deserts_vegetation_change\strz_subset.shp'
dstDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover'
#dstDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover_v3'
#dstDir = r'S:\sturt\landsat\landsat_seasonal_surface_reflectance'

# Read in shapefile and get bounding box
basename = os.path.basename(polygon).replace(r'.shp', '')
ds = ogr.Open(polygon)
for lyr in ds:
    (xmin, xmax, ymin, ymax) = lyr.GetExtent()
bbox = [int(round(xmin)), int(round(ymax)), int(round(xmax)), int(round(ymin))]
ds = None

# Construct dateList for all seasonal dates
#start = 198712198802
#end = 202403202405

start = 202406202408
end = 202409202411

dateList = []
for y1 in range(1987, 2025):
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
#srcDir = r'/vsicurl/https://data.tern.org.au/rs/public/data/landsat/seasonal_fractional_cover/fractional_cover/aus/'
#srcDir = r'/vsicurl/https://data.tern.org.au/rs/public/data/landsat/seasonal_fractional_cover_v3/fractional_cover/seasonal/nsw/'
srcDir = r'/vsicurl/https://data.tern.org.au/rs/public/data/landsat/seasonal_fractional_cover/fractional_cover/nsw/'

for date in dateList:

    #srcImage = r'lztmre_aus_m%i_dima2.vrt'%date
    srcImage = r'lztmre_nsw_m%i_dima2.tif'%date
    
    #if date <= 201303201305:
    #    srcImage = r'lztmre_nsw_m%i_dbia2.tif'%date
    #elif date >= 202112202202:
    #    srcImage = r'lzolre_nsw_m%i_dbia2.tif'%date
    #else:
    #    srcImage = r'l8olre_nsw_m%i_dbia2.tif'%date
    
    srcFile = os.path.join(srcDir, srcImage)
    dstFile = os.path.join(dstDir, srcImage.replace('nsw', 'aus').replace(r'.tif', r'_subset.tif'))
    #dstFile = os.path.join(dstDir, srcImage.replace(r'.vrt', r'_subset.tif'))
        
    if os.path.exists(dstFile) is False:
        print(srcFile)
        src_ds = gdal.Open(srcFile)
        dst_ds = gdal.Translate(dstFile, src_ds, projWin=bbox)
        dst_ds = None
        src_ds = None 

print('Image subsets downloaded')