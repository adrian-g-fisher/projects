#!/usr/bin/env python
"""
Extracts the mean and standard deviation of each fractional cover component
(green, dead and bare) for four areas.
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


def getPixelValues(info, inputs, outputs, otherargs):
    """
    Called from RIOS which reads in the image in small tiles so it is more memory efficient.
    Extracts pixel values from within polygons and stores them in a list with the date.
    It ignores any pixels that have nodata (mainly due to clouds obscuring the ground).
    """
    sites = inputs.sites[0]
    fc = inputs.fc
    fc = np.where(fc >= 100, fc - 100, 0)
    fc[inputs.fc == 0] = 255
    sitesPresent = np.unique(sites[sites != 0])
    if len(sitesPresent) > 0:
        uids = sites[sites != 0]
        bare = fc[0][sites != 0]
        green = fc[1][sites != 0]
        dead = fc[2][sites != 0]
        for i in range(uids.size):
            if bare[i] != 255:
                otherargs.pixels.append([uids[i], bare[i], green[i], dead[i]])


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
    infiles.sites = polyfile
    infiles.fc = imagefile
    otherargs.pixels = []
    applier.apply(getPixelValues, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    # Calculate statistics on pixels within polygons
    values = np.array(otherargs.pixels).astype(np.float32)
    if values.size > 0:
        uids = np.unique(values[:, 0])
        countValues = ndimage.sum(np.ones_like(values[:, 1]), values[:, 0], uids)
        meanBare = ndimage.mean(values[:, 1], values[:, 0], uids)
        stdBare = ndimage.standard_deviation(values[:, 1], values[:, 0], uids)
        meanGreen = ndimage.mean(values[:, 2], values[:, 0], uids)
        stdGreen = ndimage.standard_deviation(values[:, 2], values[:, 0], uids)
        meanDead = ndimage.mean(values[:, 3], values[:, 0], uids)
        stdDead = ndimage.standard_deviation(values[:, 3], values[:, 0], uids)
        date = int(os.path.basename(imagefile).split(r'_')[2][1:])
        
        
        
        # Write to csv
        with open(csvfile, "a") as f:
            for i in range(uids.size):
                if uids[i] == 1: name = "Quinyambie"
                if uids[i] == 2: name = "Strzelecki Regional Reserve"
                if uids[i] == 3: name = "Winnathee"
                if uids[i] == 4: name = "Sturt National Park"
                f.write('%i,%i,%s,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(date, uids[i], name, countValues[i],
                                                                       meanBare[i], stdBare[i],
                                                                       meanGreen[i], stdGreen[i],
                                                                       meanDead[i], stdDead[i],))


# Inputs and outputs
polyfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\dingo_fence\four_area_analysis\StrzStudyAreas.shp'
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\dingo_fence\four_area_analysis\StrzStudyAreas_updated.csv'
imageDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover'
imageList = glob.glob(os.path.join(imageDir, r'*.tif'))

# Write the csvfile header 
with open(csvfile, 'w') as f:
    f.write('Date,Id,Name,pixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')

# Iterate over images and get pixel values
for imagefile in imageList:
    print(os.path.basename(imagefile))
    extract_pixels(polyfile, imagefile, csvfile)

print('Pixels extracted')
