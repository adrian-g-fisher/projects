#!/usr/bin/env python
"""
Extracts the mean and standard deviation of each fractional cover component
(green, dead and bare) for each site polygon in the input shapefile. The
shapefile needs to have an attribute called "Id" which has a unique integer for
each site.
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
    Called from RIOS which reads in the image in small tiles so it is more
    memory efficient. Extracts pixel values from within polygons and stores them
    in a list with the date. It ignores any pixels that have nodata (mainly due 
    to clouds obscuring the ground).
    """
    sites = inputs.sites[0]
    fc = inputs.fc.astype(np.float32)
    sitesPresent = np.unique(sites[sites != 0])
    if otherargs.subtract == True:
        s = 100
        nodata = 0
    else:
        s = 0
        nodata = 255
    if len(sitesPresent) > 0:
        uids = sites[(sites != 0) & (fc[0] != nodata)]
        bare = fc[0][(sites != 0) & (fc[0] != nodata)] - s
        green = fc[1][(sites != 0) & (fc[0] != nodata)] - s
        dead = fc[2][(sites != 0) & (fc[0] != nodata)] - s
        
        bare[bare < 0] = 0
        green[green < 0] = 0
        dead[dead < 0] = 0
        
        bare[bare > 100] = 100
        green[green > 100] = 100
        dead[dead > 100] = 100
        
        for i in range(uids.size):
            otherargs.pixels.append([uids[i], bare[i], green[i], dead[i]])


def extract_pixels(polyfile, imagefile, csvBase, subtract):
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
    otherargs.subtract = subtract
    applier.apply(getPixelValues, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)
    
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
        for i in range(uids.size):
            siteID = int(uids[i])
            csvfile = csvBase%siteID
            with open(csvfile, "a") as f:
                f.write('%i,%i,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(date,
                                                      uids[i], countValues[i],
                                                      meanBare[i], stdBare[i],
                                                      meanGreen[i], stdGreen[i],
                                                      meanDead[i], stdDead[i],))

# Inputs and outputs
polyfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\WildDeserts_monitoringsitehectareplots.shp'

csvBase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\timeseries_fc\%s_fc_timeseries.csv'
imageDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover'
subtract = True

#csvBase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\timeseries_fc_v3\%s_fc_timeseries.csv'
#imageDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover_v3'
#subtract = False

#csvBase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\timeseries_fc_AZN\%s_fc_timeseries.csv'
#imageDir = r'S:\sturt\landsat\landsat_seasonal_fractional_cover_AZN'
#subtract = False

imageList = glob.glob(os.path.join(imageDir, r'*.tif'))

# Write the csvfile header
for i in range(1, 71):
    csvfile = csvBase%i
    with open(csvfile, 'w') as f:
        f.write('Date,Id,pixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')

# Iterate over images and get pixel values
for imagefile in imageList:
    subtract = True
    extract_pixels(polyfile, imagefile, csvBase, subtract)

print('Pixels extracted')