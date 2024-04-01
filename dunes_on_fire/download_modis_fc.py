#!/usr/bin/env python
"""
This downloads MODIS fractiuonal cover from CSIRO
"""

import os
import sys
import glob
import urllib3
import numpy as np


# Construct dateList for desired temporal extent
startDate = 200101
endDate = 202401
dateList = []
for y in range(int(str(startDate)[0:4]), int(str(endDate)[0:4])+1):
    for m in range(1, 13):
        d = (y * 100) + m
        if d >= startDate and d <= endDate:
            dateList.append(d)

#Inputs Outputs
dstDir = r'S:\aust\modis_fractional_cover'
srcDir = r'https://eo-data.csiro.au/remotesensing/v310/australia/monthly/cover/'

#Set up urllib3
http = urllib3.PoolManager()

#Iterate through dates and save the tif files
for d in dateList:
    srcImage = r'FC.v310.MCD43A4.A%i.%02d.aust.006.tif'%(int(str(d)[0:4]), int(str(d)[4:6]))
    srcFile = os.path.join(srcDir, srcImage)
    dstFile = os.path.join(dstDir, srcImage)
    if os.path.exists(dstFile) is False:
        r = http.request('GET', srcFile, preload_content=False)
        with open(dstFile,'wb') as out:
            while True:
                data = r.read(2**16)
                if not data:
                    break
                out.write(data)
        r.release_conn()

print('Grids downloaded')   
