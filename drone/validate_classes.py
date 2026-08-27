#!/usr/bin/env python
"""
This extracts classifiction values for 178 quadrat polygons across 18 drone images
- iterates over each drone image, and extracts values for polygons
- creates a CSV file with the copunt of each pixel class in each polygon

Data:
- C:/Data/grazing_study_drone_data/vectors/quadrats_classes_2026.shp
- C:/Data/grazing_study_drone_data/classified_rasters/site_date_mosaic_classified.tif

The pixel values in classified images are:
0   Bare
1   Perennial vegetation (shrubs and Mitchell Grass)
2   Low vegetation (sclerolaena, forbs, grasses)
255 Nodata
"""

import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier


def extractPixels(info, inputs, outputs, otherargs):
    """
    Gets stats
    """
    sites = inputs.sites[0]
    if np.max(sites) > 0:
        classes = inputs.classes[0]
        siteValues = sites[sites > 0]
        classValues = classes[sites > 0]
        for n in range(siteValues.size):
            otherargs.results.append([siteValues[n], classValues[n]])


def pixels_in_polygons(polyfile, imagefile, results):
    """
    This sets up RIOS to extract pixel statistics.
    """
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("Id")
    controls.setReferenceImage(imagefile)
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    infiles.sites = polyfile
    infiles.classes = imagefile
    otherargs.results = results
    applier.apply(extractPixels, infiles, outfiles, otherArgs=otherargs, controls=controls)
    return otherargs.results


# Get the right images
imageDir = 'C:/Data/grazing_study_drone_data/classified_rasters'
imageList = glob.glob(os.path.join(imageDir, '*_mosaic_classified.tif'))
siteList = np.array([os.path.basename(i).split('_')[0][0] for i in imageList])
yearList = np.array([int(os.path.basename(i).split('_')[1][0:4]) for i in imageList])
imageList = np.array(imageList)
imagesTodo = imageList[(yearList == 2026) &
                       (siteList != 'w') &
                       (imageList != os.path.join(imageDir, 'fc5_20260518_mosaic_classified.tif'))]

# Create output file and header
csvFile = 'C:/Data/grazing_study_drone_data/vectors/quadrat_classes_2026.csv'
with open(csvFile, 'w') as f:
    f.write('Id,bare_pixels,shrub_pixels,lowveg_pixels\n')

# Loop over images, using RIOS to extract pixel values for each polygon
polyfile = 'C:/Data/grazing_study_drone_data/vectors/quadrat_classes_2026.shp'
results = []
for image in imagesTodo:
    print(image)
    results = pixels_in_polygons(polyfile, image, results)

# Calculate majority class, number of majority pixels, and total number of pixels
results = np.array(results)
idValues = results[:, 0]
pixelValues = results[:, 1]
uniqueIds = np.unique(idValues)

with open(csvFile, 'a') as f:
    for i in uniqueIds:
        p = pixelValues[idValues == i]
        p0 = np.sum(p == 0)
        p1 = np.sum(p == 1)
        p2 = np.sum(p == 2)
        line = '%i,%i,%i,%i\n'%(i, p0, p1, p2)
        f.write(line)