#!/usr/bin/env python
"""

This extracts fractional cover v3 data for the star transects at Fowlers Gap

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

"""
Extracts the mean and standard deviation of each fractional cover component
(green, dead and bare) for each site polygon in the input shapefile. The
shapefile needs to have an attribute called "Id" which has a unique integer for
each site. It should also be in the same coordinate reference system as the
image data, which is EPSG:3577, or in ArcGIS this is defined as
GDA_1994_Australia_Albers.
"""


def getSuroundingPixels(info, inputs, outputs, otherargs):
    """
    Gets stats from the 8 pixels surrounding a certain site or sites 
    """
    sites = inputs.sites[0]
    for idvalue in otherargs.idstoget:
        singlesite = (sites == idvalue)
        singlesite = ndimage.maximum_filter(singlesite, size=3)
        fc = inputs.fc
        nodataPixels = (fc[0] == 255)
        bare = fc[0][singlesite]
        green = fc[1][singlesite]
        dead = fc[2][singlesite]
        nodataPixels = np.sum(nodataPixels[singlesite])
        with open(otherargs.csvfile, 'a') as f:
            line = '%i,%s'%(idvalue, otherargs.imagename)
            if nodataPixels == 0:
                line = '%s,%i'%(line, bare.size)
                line = '%s,%.2f,%.2f'%(line, np.mean(bare), np.std(bare))
                line = '%s,%.2f,%.2f'%(line, np.mean(green), np.std(green))
                line = '%s,%.2f,%.2f\n'%(line, np.mean(dead), np.std(dead))
            else:
                line = '%s,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999\n'%line
            f.write(line)


def extract_pixels(pointfile, imagefile, csvfile, idstoget):
    """
    This sets up RIOS to extract pixel statistics for the points.
    """
    # Extract the pixels
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("Id")
    controls.setReferenceImage(imagefile)
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    controls.setWindowXsize(1200)
    controls.setWindowYsize(1200)
    infiles.sites = pointfile
    infiles.fc = imagefile
    otherargs.idstoget = idstoget
    otherargs.csvfile = csvfile
    otherargs.imagename = os.path.basename(imagefile)
    applier.apply(getSuroundingPixels, infiles, outfiles,
                  otherArgs=otherargs, controls=controls)


# Read in dates for points
ids = []
sites = []
dates = []
with open("fg_star_transects.csv", 'r') as f:
    f.readline()
    for line in f:
        ids.append(int(line.strip().split(',')[0]))
        sites.append(line.strip().split(',')[1])
        dates.append(line.strip().split(',')[2])
ids = np.array(ids)
sites = np.array(sites)
dates = np.array(dates)
seasonalDates = []
for i, d in enumerate(dates):
    year = int(d[0:4])
    month = int(d[4:6])
    if month in [1, 2]:
        s = '%s12%s02'%(year-1, year)
    elif month in [3, 4, 5]:
        s = '%s03%s05'%(year, year)
    elif month in [6, 7, 8]:
        s = '%s06%s08'%(year, year)
    elif month in [9, 10, 11]:
        s = '%s09%s11'%(year, year)
    elif month == 12:
        s = '%s12%s02'%(year, year+1)
    seasonalDates.append(s)
seasonalDates = np.array(seasonalDates)

# Get imageList
imageDir = r'S:\fowlers_gap\imagery\landsat\seasonal_fractional_cover_v3'
imageList = []
idList = []
for d in np.unique(seasonalDates):
    imageList.append(os.path.join(imageDir,
                                  'lztmre_nsw_m%s_dp1a2_subset.tif'%d))         
    idList.append(ids[seasonalDates == d])

# Write the csvfile header
csvfile = r'seasonal_fc_v3_extract.csv'
with open(csvfile, 'w') as f:
    f.write('Id,image,pixels,'+
            'meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')

# Iterate over images and get pixel values
pointFile = (r'C:\Users\Adrian\OneDrive - UNSW\Documents\Fowlers_Gap\star_transects\fg_star_transects_albers.shp')
for i, imagefile in enumerate(imageList):
    extract_pixels(pointFile, imagefile, csvfile, idList[i])