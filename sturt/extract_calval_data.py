#!/usr/bin/env python
"""
Extracts the mean Landsat seasonal reflectance band data for each site polygon
in the input shapefile. The shapefile needs to have an attribute called "Id"
which has a unique integer for each site.

The output is a CSV file with rows for each site, and the following columns:
id,site,date,living,dead,bare,b1,b2,b3,b4,b5,b6

Drone imagery was  captured March-May (Autumn) in 2018, 2019, 2020, 2021, 2022

"""
import os
import sys
import numpy as np
from osgeo import ogr
from rios import applier
from scipy import ndimage


###########################
# 1. Read in the drone data
###########################

# Read in the shapefile attributes to match Id to IDENT
polyfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\WildDeserts_monitoringsitehectareplots.shp'
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(polyfile, 0)
layer = dataSource.GetLayer()
Id2Ident = {}
for feature in layer:
    Id = int(feature.GetField("Id"))
    Ident = feature.GetField("Ident")
    Id2Ident[Id] = Ident
layer.ResetReading()

# Read in drone data
droneData = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\FinalDeadAlive2018to2022.csv'
droneIdent = []
droneYear = []
droneType = []
dronePix = []
with open(droneData, 'r') as f:
    f.readline()
    for line in f:
        droneIdent.append(line.split(',')[5])
        droneYear.append(int(line.split(',')[1]))
        droneType.append(line.split(',')[7])
        dronePix.append(float(line.split(',')[6]))
droneIdent = np.array(droneIdent)
droneYear = np.array(droneYear)
droneType = np.array(droneType)
dronePix = np.array(dronePix)

# Re-arrange data and calculate living and dead proportions without shadows
dIdent = []
dYear = []
dAlive = []
dDead = []
years = np.unique(droneYear)
idents = np.unique(droneIdent)
fudgefactor = 1.00 # Proportion of PV that should be NPV
for y in years:
    for i in idents:
        if np.size(dronePix[(droneType == 'Alive') & (droneYear == y) & (droneIdent == i)]) > 0:
            alive = dronePix[(droneType == 'Alive') & (droneYear == y) & (droneIdent == i)][0]
            dead = dronePix[(droneType == 'Dead') & (droneYear == y) & (droneIdent == i)][0]
            background = dronePix[(droneType == 'Background') & (droneYear == y) & (droneIdent == i)][0]
            total = alive + dead + background
            alivePercent = 100 * alive / float(total)
            deadPercent = 100 * dead / float(total)
            alive_component = alivePercent * fudgefactor
            dead_component = (alivePercent * (1 - fudgefactor)) + deadPercent
            dIdent.append(i)
            dYear.append(y)        
            dAlive.append(alive_component)
            dDead.append(dead_component)
dIdent = np.array(dIdent)
dYear = np.array(dYear)
dAlive = np.array(dAlive)
dDead = np.array(dDead)

#############################
# 2. Extract the landsat data
#############################

def getPixelValues(info, inputs, outputs, otherargs):
    """
    Called from RIOS. Extracts pixel values from within polygons and stores them
    in a list with the date. It ignores any pixels that have nodata.
    """
    sites = inputs.sites[0]
    sr = inputs.sr.astype(np.float32)
    sitesPresent = np.unique(sites[sites != 0])
    nodata = 32767
    if len(sitesPresent) > 0:
        uids = sites[sites != 0]
        b1 = sr[0][sites != 0]
        b2 = sr[1][sites != 0]
        b3 = sr[2][sites != 0]
        b4 = sr[3][sites != 0]
        b5 = sr[4][sites != 0]
        b6 = sr[5][sites != 0]
        for i in range(uids.size):
            if b1[i] != nodata:
                otherargs.pixels.append([uids[i], b1[i], b2[i], b3[i],
                                                  b4[i], b5[i], b6[i]])


def extract_pixels(polyfile, imagefile, outCsv):
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
    infiles.sr = imagefile
    otherargs.pixels = []
    applier.apply(getPixelValues, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)
    
    # Calculate statistics on pixels within polygons
    values = np.array(otherargs.pixels).astype(np.float32)
    if values.size > 0:
        uids = np.unique(values[:, 0])
        band1 = ndimage.mean(values[:, 1], values[:, 0], uids)
        band2 = ndimage.mean(values[:, 2], values[:, 0], uids)
        band3 = ndimage.mean(values[:, 3], values[:, 0], uids)
        band4 = ndimage.mean(values[:, 2], values[:, 0], uids)
        band5 = ndimage.mean(values[:, 2], values[:, 0], uids)
        band6 = ndimage.mean(values[:, 2], values[:, 0], uids)
        date = int(os.path.basename(imagefile).split(r'_')[2][1:])
        year = int(str(date)[0:4])

        # Write to csv
        for i in range(uids.size):
            siteID = int(uids[i])
            ident = Id2Ident[siteID]

            print(siteID, ident, year)

            if dAlive[(dIdent == ident) & (dYear == year)].size > 0:
                droneAlive = dAlive[(dIdent == ident) & (dYear == year)][0]
                droneDead = dDead[(dIdent == ident) & (dYear == year)][0]
                droneBare = 100 - (droneAlive + droneDead) 

                with open(outCsv, "a") as f:
                    line = '%i,%s,%i,%.4f,%.4f,%.4f'%(siteID, ident, date,
                                                      droneAlive, droneDead,
                                                      droneBare)
                    line = '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(line,
                                                                 band1[i], band2[i],
                                                                 band3[i], band4[i],
                                                                 band5[i], band6[i])
                    f.write(line)

# Imput images
imageDir = r'S:\sturt\landsat\landsat_seasonal_surface_reflectance'
imageList = ['l8olre_aus_m201803201805_dbia2_subset.tif',
             'l8olre_aus_m201903201905_dbia2_subset.tif',
             'l8olre_aus_m202003202005_dbia2_subset.tif',
             'l8olre_aus_m202103202105_dbia2_subset.tif',
             'lzolre_aus_m202203202205_dbia2_subset.tif']
imageList = [os.path.join(imageDir, i) for i in imageList]

# Write the csvfile header
outCsv = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\sturt_calval.csv'
with open(outCsv, 'w') as f:
    f.write('id,site,date,living,dead,bare,b1,b2,b3,b4,b5,b6\n')

# Iterate over images and get pixel values
for imagefile in imageList:
    subtract = True
    extract_pixels(polyfile, imagefile, outCsv)

print('Pixels extracted')