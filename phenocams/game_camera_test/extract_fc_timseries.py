#!/usr/bin/env python
"""
This extracts Sentinel-2 fractional cover times series for the phenocams.
It extracts the mean and standard deviation of each fractional cover component
(green, dead and bare) for the 3 x 3 pixels at each point location.
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
        singlesite = ndimage.maximum_filter(singlesite, size=3)
        fc = inputs.fc
        nodataPixels = (fc[0] == 255)
        bare = fc[0][singlesite]
        green = fc[1][singlesite]
        dead = fc[2][singlesite]
        nodataPixels = np.sum(nodataPixels[singlesite])
        with open(otherargs.csvfile, 'a') as f:
            siteName = id2site[idvalue]
            line = '%s,%s'%(siteName, otherargs.imageDate)
            if nodataPixels == 0:
                line = '%s,%.2f,%.2f'%(line, np.mean(bare), np.std(bare))
                line = '%s,%.2f,%.2f'%(line, np.mean(green), np.std(green))
                line = '%s,%.2f,%.2f\n'%(line, np.mean(dead), np.std(dead))
            else:
                line = '%s,999,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999\n'%line
            f.write(line)


def extract_pixels(pointfile, imagefile, csvfile):
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
    controls.setWindowXsize(2800)
    controls.setWindowYsize(2800)
    infiles.sites = pointfile
    infiles.fc = imagefile
    otherargs.csvfile = csvfile
    otherargs.imageDate = os.path.basename(imagefile).split('_')[2]
    applier.apply(getPixels, infiles, outfiles,
                  otherArgs=otherargs, controls=controls)


# Write the csvfile header
csvfile = r'pheno_s2_fc_extract.csv'
with open(csvfile, 'w') as f:
    f.write('site,date,'+
            'meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')

# Read in shapefile and make Id-site dictionary
pointFile = (r'S:\fowlers_gap\adrian_fisher\pheno\pheno_sites.shp')
driver = ogr.GetDriverByName('ESRI Shapefile')
dataSource = driver.Open(pointFile, 1)
layer = dataSource.GetLayer()
layerDefinition = layer.GetLayerDefn()
id2site = {}
for feat in layer:
    idvalue = feat.GetField("Id")
    siteName = feat.GetField("site").lower()
    id2site[idvalue] = siteName
    feat.Destroy()
dataSource.Destroy()

# Get imageList without cloudy images
cloudyDates = ['20200620', '20200627', '20200705', '20200710', '20200712',
               '20200806', '20200809', '20200811', '20200814', '20200816',
               '20200821', '20200824', '20200908', '20200918', '20200920',
               '20200925', '20201005', '20201008', '20201023', '20201025',
               '20201028', '20201030', '20201104', '20201122', '20201129',
               '20201214', '20201219', '20201227', '20210126', '20210128',
               '20210101', '20210103', '20210108', '20210205', '20210207',
               '20210212']
imageDir = r'S:\fowlers_gap\imagery\sentinel2\fractional_cover'
imageList = []
for imageFile in glob.glob(os.path.join(imageDir, "*.img")):
    imageDate = os.path.basename(imageFile).split('_')[2]
    if imageDate not in cloudyDates:
        imageList.append(imageFile)

# Iterate over images and get pixel values
for imageFile in imageList:
    extract_pixels(pointFile, imageFile, csvfile)