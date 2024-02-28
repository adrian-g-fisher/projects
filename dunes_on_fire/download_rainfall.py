#!/usr/bin/env python
"""

This downloads BOM precipitation grids of Australia from NCI

"""

#Import packages
import os
import sys
import glob
import urllib3
import numpy as np


# Construct dateList for desired temporal extent  
start = 2021
end = 2022
dateList = range(start, end+1)

#Inputs Outputs

dstDir = r'S:\aust\bom_climate_grids\bom_monthly_rainfall\1second\NetCDF'
dstDir = r'S:\aust\bom_climate_grids\bom_monthly_rainfall\1second\NetCDF'
#srcDir = r'https://dapds00.nci.org.au/thredds/fileServer/zv2/agcd/v2/precip/total/r001/01month/'
srcDir = r'https://dapds00.nci.org.au/thredds/fileServer/zv2/agcd/v2-0-1/precip/total/r001/01month/'

#Set up urllib3

http = urllib3.PoolManager()

#Iterate through dates and save the NetCDF

for date in dateList:
    srcImage = r'agcd_v2-0-1_precip_total_r001_monthly_%i.nc'%date
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


