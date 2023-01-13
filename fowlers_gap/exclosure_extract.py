#!/usr/bin/env python
"""

This extracts fractional cover data for inside and outside exclosures at Fowlers
Gap.

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


def getPixels(info, inputs, outputs, otherargs):
    """
    Gets stats from the 8 pixels surrounding a certain site or sites 
    """
    sites = inputs.sites[0]
    for idvalue in np.unique(sites[sites != 0]):
        singlesite = (sites == idvalue)
        fc = inputs.fc
        nodataPixels = (fc[0] == 0)
        fc = np.where(fc >= 100, fc - 100, 0)
        bare = fc[0][singlesite]
        green = fc[1][singlesite]
        dead = fc[2][singlesite]
        nodata = nodataPixels[singlesite]
        bare = bare[nodata == 0]
        green = green[nodata == 0]
        dead = dead[nodata == 0]
        with open(otherargs.csvfile, 'a') as f:
            line = '%i,%s'%(idvalue, otherargs.date)
            if bare.size > 0:
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


def extract_pixels(polyfile, imagefile, csvfile):
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
    controls.setWindowXsize(4500)
    controls.setWindowYsize(4500)
    infiles.sites = polyfile
    infiles.fc = imagefile
    otherargs.csvfile = csvfile
    otherargs.date = os.path.basename(imagefile).split('_')[2][1:]
    applier.apply(getPixels, infiles, outfiles,
                  otherArgs=otherargs, controls=controls)


# Get imageList
imageDir = r'S:\fowlers_gap\imagery\landsat\seasonal_fractional_cover'
imageList = glob.glob(os.path.join(imageDir, "*.tif"))

# Write the csvfile header
csvfile = r'exclosure_seasonal_fc.csv'
with open(csvfile, 'w') as f:
    f.write('Id,date,pixels,'+
            'meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')

# Iterate over images and get pixel values
polyFile = (r'C:\Users\Adrian\OneDrive - UNSW\Documents\student_projects\mphil\matt_smith\data\Fowl_treat_polys_albers.shp')
for imagefile in imageList:
    extract_pixels(polyFile, imagefile, csvfile)