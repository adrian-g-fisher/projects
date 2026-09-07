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
gdal.UseExceptions()


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


def extract_validation():
    
    # Get the right images
    imageDir = 'C:/Data/grazing_study_drone_data/classified_rasters'
    imageList = glob.glob(os.path.join(imageDir, '*_mosaic_classified.tif'))
    siteList = np.array([os.path.basename(i).split('_')[0][0] for i in imageList])
    yearList = np.array([int(os.path.basename(i).split('_')[1][0:4]) for i in imageList])
    imageList = np.array(imageList)
    imagesTodo = imageList[(yearList == 2026) & (siteList != 'w')]

    # Get attributes from quadrats (Id, name, class, descriptio)
    polyfile = 'C:/Data/grazing_study_drone_data/vectors/quadrat_classes_2026.shp'
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(polyfile, 0)
    layer = dataSource.GetLayer()
    attributes = []
    for feature in layer:
        Id = feature.GetField("Id")
        name = feature.GetField("name")
        clas = feature.GetField("class")
        descriptio = feature.GetField("descriptio")
        attributes.append([Id, name, clas, descriptio])
    layer.ResetReading()
    attributes = np.array(attributes)

    # Create output file and header
    csvFile = 'C:/Data/grazing_study_drone_data/vectors/quadrat_classes_2026.csv'
    with open(csvFile, 'w') as f:
        f.write('Id,name,class,description,pixels,barePercent,shrubPercent,lowvegPercent,dominant\n')

    # Loop over images, using RIOS to extract pixel values for each polygon
    results = []
    for image in imagesTodo:
        print(image)
        results = pixels_in_polygons(polyfile, image, results)

    # Calculate pixel counts, error matrix, and write to CSV
    results = np.array(results)
    idValues = results[:, 0]
    pixelValues = results[:, 1]
    uniqueIds = np.unique(idValues)
    errorMatrix = np.zeros((3, 3), dtype=np.uint8)
    with open(csvFile, 'a') as f:
        for i in uniqueIds:
            line = ','.join(attributes[(attributes[:, 0] == str(i)), :].tolist()[0])
            p = pixelValues[idValues == i]
            pixels = float(p.size)
            pixelsBare = np.sum(p == 0)
            pixelsShrub = np.sum(p == 1)
            pixelsLowveg = np.sum(p == 2)
            barePercent = 100 * (pixelsBare/pixels)
            shrubPercent = 100 * (pixelsShrub/pixels)
            lowvegPercent = 100 * (pixelsLowveg/pixels)
            
            if barePercent >= 50:
                dominant = 'bare'
            elif shrubPercent >= 50:
                dominant = 'perennial vegetation'
            else:
                dominant = 'low vegetation'
            
            reference = line.split(',')[2]
            if reference in ['shrub', 'mitchell grass']:
                reference = 'perennial vegetation'
                
            r2i = {'bare': 0, 'low vegetation' : 1, 'perennial vegetation' : 2}
            for r in ['bare', 'low vegetation', 'perennial vegetation']:
                for c in ['bare', 'low vegetation', 'perennial vegetation']:
                    if reference == r and dominant == c:
                        errorMatrix[r2i[r], r2i[c]] += 1
            
            line = '%s,%i,%.2f,%.2f,%.2f,%s\n'%(line, pixels, barePercent, shrubPercent, lowvegPercent, dominant)
            f.write(line)
            return errorMatrix


def analyse_validation():
    
    # Read in validation CSV
    
    # Create box plots comparing:
    # barePercent of bare and not bare
    # shrubPercent of perennial veg and not perennial veg
    # lowvegPercent of low vegetation and not low vegetation
    
    

#errorMatrix = extract_validation()
#print(errorMatrix)
# 22 14  1
#  0 50 34
#  0  7 50
analyse_validation()

