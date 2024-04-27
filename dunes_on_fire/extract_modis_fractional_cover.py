#!/usr/bin/env python
"""
"""
import os, sys, glob
import numpy as np
from osgeo import ogr
from rios import applier
from scipy import ndimage

def getPixelValues(info, inputs, outputs, otherargs):
    """
    Called from RIOS which reads in the image in small tiles so it is more
    memory efficient. Extracts pixel values from within polygons and stores them
    in a list with the date. It ignores any pixels that have nodata (mainly due 
    to clouds obscuring the ground).
    """
    poly = inputs.poly[0]
    fc = inputs.fc.astype(np.float32)
    polysPresent = np.unique(poly[poly != 0])
    nodata = 255
    if len(polysPresent) > 0:
        uids = poly[(poly != 0) & (fc[0] != nodata)]
        bare = fc[0][(poly != 0) & (fc[0] != nodata)]
        green = fc[1][(poly != 0) & (fc[0] != nodata)]
        dead = fc[2][(poly != 0) & (fc[0] != nodata)]
        
        bare[bare < 0] = 0
        green[green < 0] = 0
        dead[dead < 0] = 0
        
        bare[bare > 100] = 100
        green[green > 100] = 100
        dead[dead > 100] = 100
        
        if len(uids) > 0:
            for i in range(uids.size):
                otherargs.pixels.append([uids[i], bare[i], green[i], dead[i]])


def extract_fc(shapefile):
    """
    Uses RIOS to extract monthly MODIS fractional cover for dunefield polygons.
    """
    outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\fractionalcover'
    
    # Read in ID values and Names from shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    ID2Name = {}
    for feature in layer:
        ID = int(feature.GetField("Id"))
        Name = feature.GetField("Code")
        ID2Name[ID] = Name
    layer.ResetReading()
    n = len(ID2Name)
    
    # Iterate over ID values creating csv files to save results
    for ID in range(1, n+1):
        Name = ID2Name[ID]
        outfile = os.path.join(outdir, 'fractionalcover_%s.csv'%(Name))
        with open(outfile, 'w') as f:
            f.write('Date,ID,Pixel_count,BS_mean,BS_std,PV_mean,PV_std,NPV_mean,NPV_std\n')
    
    # Iterate over FC images
    for imagefile in glob.glob(r"S:\aust\modis_fractional_cover\*.tif"):
        year = imagefile.replace(r".tif", "").split(r".")[-4][1:]
        month = imagefile.replace(r".tif", "").split(r".")[-3]
        date = year + month
        
        print(date)
        
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        controls.setBurnAttribute("Id")
        infiles.fc = imagefile
        infiles.poly = shapefile
        otherargs.pixels = []
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
    
            # Write to csv
            for i in range(uids.size):
                siteID = int(uids[i])
                Name = ID2Name[siteID]
                outfile = os.path.join(outdir, 'fractionalcover_%s.csv'%(Name))
                with open(outfile, "a") as f:
                f.write('%s,%i,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(date,
                                                      siteID, countValues[i],
                                                      meanBare[i], stdBare[i],
                                                      meanGreen[i], stdGreen[i],
                                                      meanDead[i], stdDead[i]))


# Get MODIS data for each polygon
shapefile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Australian_Dunefields_202402\dunefields_geographic_multipart.shp'
extract_fc(shapefile)