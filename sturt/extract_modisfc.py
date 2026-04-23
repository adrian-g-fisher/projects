#!/usr/bin/env python
"""
This extracts time series MODIS fractional cover data for polygons. It
calculates the mean and standard deviation values for monthly images each 
component (green, dead and bare) and saves it to a CSV file.

The poygon shapefile needs to have an integer attribute called Id and should be
in the same projection as the MODIS imagery (WGS84). MODIS data for
Australia are located in S:/aust/modis_fractional_cover
"""


import os
import sys
import glob
import math
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import rat
from scipy import ndimage
from datetime import datetime
ogr.UseExceptions()


def getFC(info, inputs, outputs, otherargs):
    """
    Gets fractional cover stats
    """
    sites = inputs.sites[0]
    for idvalue in np.unique(sites[sites != 0]):
        singlesite = (sites == idvalue)
        fc = inputs.fc
        nodataPixels = (fc[0] == 255)
        bare = fc[0][singlesite]
        green = fc[1][singlesite]
        dead = fc[2][singlesite]
        nodata = nodataPixels[singlesite]
        bare = bare[nodata == 0]
        green = green[nodata == 0]
        dead = dead[nodata == 0]
        
        line = []
        if bare.size > 0:
            line = [idvalue, otherargs.date, bare.size,
                    np.mean(bare), np.std(bare),
                    np.mean(green), np.std(green),
                    np.mean(dead), np.std(dead),
                    999, 999, 999, 999, 999, 999, 999]
        else:
            line = [idvalue, otherargs.date, bare.size,
                    999, 999, 999, 999, 999, 999,
                    999, 999, 999, 999, 999, 999, 999]
        
        otherargs.results.append(line)
        

def extract_fc(imagefile, results, tempFile, xSize, ySize):
    """
    This sets up RIOS to extract pixel statistics for fractional cover.
    """
    # Extract the pixels
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.sites = tempFile
    infiles.fc = imagefile
    otherargs.results = results
    year = imagefile.replace(r".tif", "").split(r".")[-4][1:]
    month = imagefile.replace(r".tif", "").split(r".")[-3]
    otherargs.date = int(year + month)    
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    controls.setReferenceImage(tempFile)
    controls.setWindowXsize(xSize)
    controls.setWindowYsize(ySize)
    applier.apply(getFC, infiles, outfiles, otherArgs=otherargs, controls=controls)
    return otherargs.results


# Define inputs and outputs
polyfile = r'C:\Users\z9803884\OneDrive - UNSW\Documents\student_projects\phd\matt_brun\polygon.shp'
csvfile = r'C:\Users\z9803884\OneDrive - UNSW\Documents\student_projects\phd\matt_brun\polygon_fc_extract.csv'

# Create temporary subset over polygon area
srcFile = r'S:\aust\modis_fractional_cover\FC.v310.MCD43A4.A2026.02.aust.061.tif'
tempFile = r'temp.tif'
ds = ogr.Open(polyfile)
for lyr in ds:
    (xmin, xmax, ymin, ymax) = lyr.GetExtent()
ds = None
src_ds = gdal.Open(srcFile)
dst_ds = gdal.Translate(tempFile, src_ds, bandList=[1], projWin=[xmin, ymax, xmax, ymin])
dst_ds = None
src_ds = None
ds = gdal.Open(tempFile, gdal.GA_Update)
band = ds.GetRasterBand(1)
zeros = np.zeros((band.YSize, band.XSize), dtype=np.uint8)
band.WriteArray(zeros)
xSize = ds.RasterXSize
ySize = ds.RasterYSize
ds = None
xSize = math.ceil(xSize/256.0) * 256
ySize = math.ceil(ySize/256.0) * 256

# Burn polygons into temp
ds = gdal.Open(tempFile, gdal.GA_Update)
ps = gdal.OpenEx(polyfile)
gdal.Rasterize(ds, ps, attribute="Id")
ds = None
ps = None

# Iterate over fractional over data
imageDir = r'S:\aust\modis_fractional_cover'
imageList = glob.glob(os.path.join(imageDir, '*.tif'))
results = []
for imagefile in imageList:
    results = extract_fc(imagefile, results, tempFile, xSize, ySize)
    print(imagefile)
results = np.array(results)
os.remove(tempFile)

# Write CSV file
with open(csvfile, 'w') as f:
    f.write('Id,date,modisPixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')
    (rows, cols) = results.shape
    for r in range(rows):
        d = results[r, :]
        line = '%i,%i,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8])
        f.write(line)