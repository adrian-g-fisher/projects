#!/usr/bin/env python
"""

This extracts seasonal NDVI for the Plains Wanderer sites using landsat surface
reflectance imagery.

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
    Gets stats from all pixels in each site polygon
    """
    sites = inputs.sites[0]
    for idvalue in np.unique(sites[sites != 0]):
        singlesite = (sites == idvalue)
        sr = inputs.sr
        nodataPixels = (sr[0] == 32767)
        red = sr[2][singlesite].astype(np.float32)
        nir = sr[3][singlesite].astype(np.float32)
        num = nir - red
        den = nir + red
        den_nozero = np.where(den == 0, 1, den)
        ndvi = num / den_nozero
        nodata = nodataPixels[singlesite]
        nodata[den == 0] = 1        
        ndvi = ndvi[nodata == 0]
        with open(otherargs.csvfile, 'a') as f:
            line = '%i,%s'%(idvalue, otherargs.date)
            if ndvi.size > 0:
                line = '%s,%i'%(line, ndvi.size)
                line = '%s,%.2f,%.2f\n'%(line, np.mean(ndvi), np.std(ndvi))
            else:
                line = '%s,999'%line
                line = '%s,999,999\n'%line
            f.write(line)


def extract_pixels(polyfile, imagefile, csvfile):
    """
    This sets up RIOS to extract pixel statistics for the polygons.
    """
    # Extract the pixels
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("Id")
    controls.setReferenceImage(imagefile)
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    controls.setWindowXsize(7000)
    controls.setWindowYsize(7000)
    infiles.sites = polyfile
    infiles.sr = imagefile
    otherargs.csvfile = csvfile
    otherargs.imagename = os.path.basename(imagefile)
    otherargs.date = os.path.basename(imagefile).split('_')[2][1:]
    print(otherargs.date)
    applier.apply(getPixels, infiles, outfiles,
                  otherArgs=otherargs, controls=controls)


# Get imageList
imageDir = r'S:\hay_plain\landsat\landsat_seasonal_surface_reflectance'
imageList = glob.glob(os.path.join(imageDir, '*.tif'))

# Write the csvfile header
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\seasonal_ndvi_extract.csv'
with open(csvfile, 'w') as f:
    f.write('Id,date,pixels,meanNDVI,stdevNDVI\n')

# Iterate over images and get pixel values
polyfile = (r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\PW_sites_2_albers.shp')
for imagefile in imageList:
    extract_pixels(polyfile, imagefile, csvfile)