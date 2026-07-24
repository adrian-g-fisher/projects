#!/usr/bin/env python
"""
This downloads global MODIS fractional cover from CSIRO

Needs the MODIS 

"""

import os
import sys
import glob
import urllib3
import numpy as np
import pandas as pd


# Get the fileList
url = "https://thredds.nci.org.au/thredds/catalog/tc43/modis-fc/v310/tiles/monthly/cover/catalog.html"
tables = pd.read_html(url)
df = tables[0]
fileList = df["Dataset"][1:].tolist()

#Inputs Outputs
dstDir = r'S:\global\modis_fractional_cover\netcdf'
srcDir = r'https://thredds.nci.org.au/thredds/fileServer/tc43/modis-fc/v310/tiles/monthly/cover/'

#Set up urllib3
http = urllib3.PoolManager()

#Iterate through fileList and save the files 
for srcImage in fileList:
    srcFile = os.path.join(srcDir, srcImage)
    dstFile = os.path.join(dstDir, srcImage)
    if os.path.exists(dstFile) is False:
        print('Downloading %s'%srcImage)
        r = http.request('GET', srcFile, preload_content=False)
        with open(dstFile,'wb') as out:
            while True:
                data = r.read(2**16)
                if not data:
                    break
                out.write(data)
        r.release_conn()
    else:
        print('%s already downloaded'%srcImage)

print('Grids downloaded')
