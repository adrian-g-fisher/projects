#!/usr/bin/env python
"""
This extracts time series raster data for the polygons across Saudi Arabia, 
calculating mean and standard deviation values for monthly images of:
 - Each fractional cover component (green, dead and bare)
 - Rainfall
 - Maximum temperature
 - Minimum temperature
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


def getClimate(info, inputs, outputs, otherargs):
    """
    Gets climate data stats
    """
    sites = inputs.sites[0]
    for idvalue in np.unique(sites[sites != 0]):
        singlesite = (sites == idvalue)
        rain = inputs.rain[0][singlesite]
        minT = inputs.minT[0][singlesite]
        maxT = inputs.maxT[0][singlesite]
        n = (otherargs.results[:,0] == idvalue) & (otherargs.results[:,1] == otherargs.date)
        otherargs.results[n,  9] = rain.size
        otherargs.results[n, 10] = np.mean(rain)
        otherargs.results[n, 11] = np.std(rain)
        otherargs.results[n, 12] = np.mean(maxT)
        otherargs.results[n, 13] = np.std(maxT)
        otherargs.results[n, 14] = np.mean(minT)
        otherargs.results[n, 15] = np.std(minT)
        

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


def extract_climate(polyfile, resampled, rainImage, maxImage, minImage, results, date):
    """
    This sets up RIOS to extract pixel statistics for the climate rasters.
    """
    # Extract the pixels
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("Id")
    controls.setReferenceImage(resampled)
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    controls.setWindowXsize(60)
    controls.setWindowYsize(60)
    controls.setResampleMethod('near')
    infiles.resampled = resampled
    infiles.sites = polyfile
    infiles.rain = rainImage
    infiles.maxT = maxImage
    infiles.minT = minImage
    otherargs.results = results
    otherargs.date = int(date)
    applier.apply(getClimate, infiles, outfiles, otherArgs=otherargs, controls=controls)
    return otherargs.results


# Put all data into a list
results = []

# Fractional over (200101-202412)
imageDir = r'C:\Data\saudi_arabia\modis\merged_tifs'
imageList = glob.glob(os.path.join(imageDir, '*.tif'))
polyfile = r'C:\Data\saudi_arabia\shapefiles\polygons_sinusoidal.shp'
for imagefile in imageList:
    results = extract_fc(polyfile, imagefile, results)
    print(imagefile, len(results))
results = np.array(results)

# Rainfall and temp (200101-202412)
polyfile = r'C:\Data\saudi_arabia\shapefiles\polygons_wgs84.shp'
resampled = r'C:\Data\saudi_arabia\cruts\resampled\resampled_clip.tif'
for y in range(2001, 2025):
    for m in range(1, 13):
        date = '%i%02d'%(y, m)
        rainImage = r'C:\Data\saudi_arabia\cruts\precipitation\wc2.1_cruts4.09_2.5m_prec_%i-%02d.tif'%(y, m)
        maxImage = r'C:\Data\saudi_arabia\cruts\temp_max\wc2.1_cruts4.09_2.5m_tmax_%i-%02d.tif'%(y, m)
        minImage = r'C:\Data\saudi_arabia\cruts\temp_min\wc2.1_cruts4.09_2.5m_tmin_%i-%02d.tif'%(y, m)
        results = extract_climate(polyfile, resampled, rainImage, maxImage, minImage, results, date)
        print(date, results.shape)

# Write CSV
csvfile = r'C:\Data\saudi_arabia\shapefiles\polygon_data_extract.csv'
with open(csvfile, 'w') as f:
    f.write('Id,date,modisPixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead,'+
            'crutsPixels,meanRain,stdevRain,meanMaxtemp,stdevMaxtemp,meanMintemp,stdevMintemp\n')

    (rows, cols) = results.shape
    for r in range(rows):
        d = results[r, :]
        line = '%i,%i,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,'%(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8])
        line = line + '%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(d[9], d[10], d[11], d[12], d[13], d[14], d[15])
        f.write(line)