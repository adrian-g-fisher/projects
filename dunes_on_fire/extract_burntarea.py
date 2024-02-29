#!/usr/bin/env python
"""
"""
import os, sys, glob
import numpy as np
from osgeo import ogr
from rios import applier


def get_burnt(info, inputs, outputs, otherargs):
    """
    Counts pixels in polygons.
    """
    burn = inputs.burnt[0]
    poly = inputs.poly[0]
    polysPresent = np.unique(poly[poly != 0])
    if len(polysPresent) > 0:
        for p in polysPresent:
            otherargs.counts[0, p-1] += np.sum((poly == p) & (burn == 1))
            otherargs.counts[1, p-1] += np.sum(poly == p)
    
def extract_burntarea(shapefile):
    """
    Uses RIOS to extract monthly burnt area for dunefield polygons.
    """
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
    
    # Iterate over ID values creating csv files to save results
    for ID in range(1, 79):
        Name = ID2Name[ID]
        outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\burntarea'
        outfile = os.path.join(outdir, 'burntarea_%s.csv'%(Name))
        with open(outfile, 'w') as f:
            f.write('ID,Date,Burnt_area_percent,Pixel_count\n')
    
    # Iterate over burnt images
    for imagefile in glob.glob(r"S:\aust\modis_burned_area\monthly_tiffs\*.tif"):
        date = imagefile.replace(r".tif", "").split(r"_")[-1]
        
        print(date)
        
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        controls.setFootprintType(applier.INTERSECTION)
        controls.setBurnAttribute("ID")
        infiles.burnt = imagefile
        infiles.poly = shapefile
        otherargs.counts = np.zeros((2, 78), dtype=np.uint64)
        applier.apply(get_burnt, infiles, outfiles, otherArgs=otherargs, controls=controls)
        
        for i, ID in enumerate(range(1, 79)):
            Name = ID2Name[ID]
            burntpixels = otherargs.counts[0, i]
            totalpixels = otherargs.counts[1, i]
            burn_percent = 100 * (burntpixels / totalpixels)
            outfile = os.path.join(outdir, 'burntarea_%s.csv'%(Name))
            with open(outfile, 'a') as f:
                f.write('%s,%s,%.2f,%i\n'%(ID, date, burn_percent, totalpixels))

# Get burnt area data
shapefile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Australian_Dunefields_202402\dunefields_sinusoidal_multipart.shp'
extract_burntarea(shapefile)