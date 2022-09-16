#!/usr/bin/env python

import os
import sys
from osgeo import gdal, ogr
from datetime import datetime

# Inputs and outputs

polygon = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\strz_subset.shp'
#dstDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover'
dstDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover_v3'

# Read in shapefile and get bounding box
basename = os.path.basename(polygon).replace(r'.shp', '')
ds = ogr.Open(polygon)
for lyr in ds:
    (xmin, xmax, ymin, ymax) = lyr.GetExtent()
bbox = [int(round(xmin)), int(round(ymax)), int(round(xmax)), int(round(ymin))]
ds = None

# Construct dateList for all seasonal dates
start = 198712198802
end = 202203202205
dateList = []
for y1 in range(1987, 2023):
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
#srcDir = r'/vsicurl/https://dap.tern.org.au/thredds/fileServer/landscapes/remote_sensing/landsat/seasonal_fractional_cover/fractional_cover/aus/'
srcDirs = [r'/vsicurl/https://dap.tern.org.au/thredds/fileServer/landscapes/remote_sensing/landsat/seasonal_fractional_cover_v3/fractional_cover/seasonal/nsw/',
           r'/vsicurl/https://dap.tern.org.au/thredds/fileServer/landscapes/remote_sensing/landsat/seasonal_fractional_cover_v3/fractional_cover/seasonal/sa/',
           r'/vsicurl/https://dap.tern.org.au/thredds/fileServer/landscapes/remote_sensing/landsat/seasonal_fractional_cover_v3/fractional_cover/seasonal/qld/']
state = ['nsw', 'sa', 'qld']

for date in dateList:

    #srcImage = r'lztmre_aus_m%i_dima2.vrt'%date
    for i in range(3):
        srcDir = srcDirs[i]
        srcImage = r'lztmre_%s_m%i_dp1a2.tif'%(state[i], date)
        srcFile = os.path.join(srcDir, srcImage)
        dstFile = os.path.join(dstDir, srcImage.replace(r'.tif', r'_subset.tif'))
        
        print(os.path.basename(dstFile))
        
        if os.path.exists(dstFile) is False:
            src_ds = gdal.Open(srcFile)
            dst_ds = gdal.Translate(dstFile, src_ds, projWin=bbox)
            dst_ds = None
            src_ds = None 

print('Image subsets downloaded')