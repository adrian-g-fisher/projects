#!/usr/bin/env python
"""

                    kwargs = {'format': 'VRT', 'bandList': [b]}
                    pvFile = outFile.replace('.tif', '_pv.vrt')
                    ds = gdal.Translate(pvFile, 'NETCDF:"%s":phot_veg'%inFile, **kwargs)
                    ds = None
                    npvFile = outFile.replace('.tif', '_npv.vrt')
                    ds = gdal.Translate(npvFile, 'NETCDF:"%s":nphot_veg'%inFile, **kwargs)
                    ds = None
                    bsFile = outFile.replace('.tif', '_bs.vrt')
                    ds = gdal.Translate(bsFile, 'NETCDF:"%s":bare_soil'%inFile, **kwargs)
                    ds = None
                    outVrt = outFile.replace('.tif', '.vrt')
                    outds = gdal.BuildVRT(outVrt, [bsFile, pvFile, npvFile], separate=True)
                    outds = gdal.Translate(outFile, outds)
                    bandnames = ['Bare', 'PV', 'NPV']
                    for i in range(3):
                        band = outds.GetRasterBand(i+1)
                        band.SetDescription(bandnames[i])
                    gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
                    outds.BuildOverviews()
                    outds = None
                    
"""
import os, sys
import numpy as np
import glob
import rioxarray
from osgeo import gdal
gdal.UseExceptions()


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

inDir = r'C:\Users\Adrian\Documents\temp\netcdf'
outDir = r'C:\Users\Adrian\Documents\temp\tif'

# Make monthly geotiffs
for sceneInDir in glob.glob(os.path.join(inDir, 'h*v*')):
    sceneOutDir = os.path.join(outDir, os.path.basename(sceneInDir))
    
    # Make output folders
    if os.path.exists(sceneOutDir) is False:
        os.mkdir(sceneOutDir)
    
    # Bad yearmonth list
    bad = ['06_2023_02', '06_2023_03', '06_2023_04', '06_2023_05', '06_2023_06', '06_2023_07', '06_2023_08', '06_2023_09', '06_2023_10', '06_2023_11', '06_2023_12',
           '61_2023_01', '61_2023_02',
           '61_2025_06', '61_2025_07', '61_2025_08', '61_2025_09', '61_2025_10', '61_2025_11', '61_2025_12']
    
    # Iterate over netcdf images and create monthly tif images
    for inFile in glob.glob(os.path.join(sceneInDir, '*.nc')):
        code = inFile[-5:-3]
        year = inFile[-11:-7]
        for b in range(1, 13):
            codeyearmonth = '%s_%s_%02d'%(code, year, b)
            if codeyearmonth not in bad:
                outFile = os.path.join(sceneOutDir, os.path.basename(inFile).replace('.006.nc', '%02d.tif'%b).replace('.061.nc', '%02d.tif'%b))
                if os.path.exists(outFile) is False:
                    
                    print(codeyearmonth)
                    
                    ds = rioxarray.open_rasterio(inFile)
                    ds = ds.sel(time="%s-%02d"%(year, b)).isel(time=0)
                    ds.rio.to_raster(outFile)

sys.exit()


