#!/usr/bin/env python
"""
This extracts time series MODIS fractional cover data for polygons. It
calculates the mean and standard deviation values for monthly images each 
component (green, dead and bare) and saves it to a CSV file.

The poygon shapefile needs to have an integer attribute called Id and should be
in the same projection as the MODIS imagery (WGS84). MODIS data for
Australia are located in S:\aust\modis_fractional_cover
"""


import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import rat
from scipy import ndimage
from datetime import datetime


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
        

def extract_fc(polyfile, imagefile, results):
    """
    This sets up RIOS to extract pixel statistics for fractional cover.
    """
    # Extract the pixels
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("Id")
    controls.setReferenceImage(imagefile)
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    controls.setWindowXsize(4800)
    controls.setWindowYsize(4800)
    infiles.sites = polyfile
    infiles.fc = imagefile
    otherargs.results = results
    otherargs.date = int(imagefile.replace(r".tif", "")[-6:])
    applier.apply(getFC, infiles, outfiles, otherArgs=otherargs, controls=controls)
    return otherargs.results

# Put all data into a list
results = []

# Fractional over (200101-202412)
imageDir = r'S:\aust\modis_fractional_cover'
imageList = glob.glob(os.path.join(imageDir, '*.tif'))
polyfile = r'polygons.shp'
for imagefile in imageList:
    results = extract_fc(polyfile, imagefile, results)
    print(imagefile, len(results))
results = np.array(results)

# Write CSV
csvfile = r'polygon_data_extract.csv'
with open(csvfile, 'w') as f:
    f.write('Id,date,modisPixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')
    (rows, cols) = results.shape
    for r in range(rows):
        d = results[r, :]
        line = '%i,%i,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8])
        f.write(line)