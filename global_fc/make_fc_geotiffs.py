#!/usr/bin/env python
"""

conda activate modis
                    
"""
import os, sys
import numpy as np
import glob
import rioxarray
from osgeo import gdal
gdal.UseExceptions()

inDir = r'S:\global\modis_fractional_cover\netcdf'
outDir = r'S:\global\modis_fractional_cover\tif'

bad_dates = [202107, 202108, 202109, 202110, 202111, 202112]

# Iterate over netcdf images and create monthly tif images
for inFile in glob.glob(os.path.join(inDir, '*.nc')):
    outBase = os.path.basename(inFile).replace('.006.nc', '').replace('.061.nc', '').replace('.', '_')
    for month in range(1, 13):
        outFile = os.path.join(outDir, '%s%02d.tif'%(outBase, month))
        yearMonth = outFile.replace('.tif', '')[-6:]
        if yearMonth not in bad_dates:
            if os.path.exists(outFile) is False:
                
                print("Creating %s"%outFile)
                
                ds = rioxarray.open_rasterio(inFile)
                #ds = ds.sel(time="%s-%02d"%(year, month)).isel(time=0)
                ds = ds.sel(band=month)
                ds.rio.write_crs("epsg:54008", inplace=True)
                ds.rio.to_raster(outFile)
                
                sys.exit()

sys.exit()

# Merge dates together
baseDir = r'C:\Users\Adrian\Documents\temp\tif'
outDir = r'C:\Users\Adrian\Documents\temp\merged_tifs'
inDirList = ['h21v06', 'h21v07', 'h22v06', 'h22v07']
for file_1 in glob.glob(os.path.join(os.path.join(baseDir, inDirList[0]), '*.tif')):
    file_2 = file_1.replace(inDirList[0], inDirList[1])
    file_3 = file_1.replace(inDirList[0], inDirList[2])
    file_4 = file_1.replace(inDirList[0], inDirList[3])
    outFile = os.path.join(outDir, os.path.basename(file_1).replace(inDirList[0], 'merged'))
    if os.path.exists(outFile) is False:
        outVrt = outFile.replace('.tif', '.vrt')
        outds = gdal.BuildVRT(outVrt, [file_1, file_2, file_3, file_4])
        outds = gdal.Translate(outFile, outds)
        bandnames = ['Photosynthetic vegetation', 'Non-photosynthetic vegetation', 'Bare soil', 'Total cover']
        for i in range(4):
            band = outds.GetRasterBand(i+1)
            band.SetDescription(bandnames[i])
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        outds.BuildOverviews()
        outds = None
        os.remove(outVrt)

    
sys.exit()




