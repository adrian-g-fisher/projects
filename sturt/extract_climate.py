#!/usr/bin/env python
"""
This extracts data from climate surfaces for each dune crest point and writes
the data to the shapefile.
"""


import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import rat
from scipy import ndimage


def calcWoody(info, inputs, outputs, otherargs):
    """
    Calculates woody cover for 50 m diameter areas around each point.
    """
    woodyMask = inputs.woody[0]
    nodataMask = np.where(woodyMask == 255, 1, 0)
    sites = inputs.sites[0]
    sitesPresent = np.unique(sites[sites != 0])
    if sitesPresent.size > 0:
        for siteId in sitesPresent:
            s = np.where(sites == siteId, 1, 0)
            s = ndimage.maximum_filter(s, footprint=disk(50))  # Dilate to 25m
            w = np.sum(woodyMask[(s == 1) & (nodataMask == 0)])# Count woody 
            t = np.sum(s[(s == 1) & (nodataMask == 0)])        # Count total
            w = 100 * (w/float(t))                             # Woody percent
            otherargs.cover.append([float(siteId), w])         # Append to list


def process_points(pointFile, woody):
    """
    Sets up RIOS to calculate the woody cover, then writes the data to the point
    shapefile.
    """
    # Set up RIOS to do the calculation
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.sites = pointFile
    infiles.woody = woody
    otherargs.cover = []    
    controls.setBurnAttribute("Id")
    controls.setVectorDatatype(np.uint16)
    controls.setOverlap(50) #25 m distance in 0.5 m pixels 
    applier.apply(calcWoody, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)
    cover = np.array(otherargs.cover)
    
    # Write the data to the shapefiles
    data_source = ogr.Open(pointFile, 1)
    layer = data_source.GetLayer(0)
    for feat in layer:
        fid = int(feat.GetField("Id"))
        if fid in cover[:, 0]:
            w = cover[:, 1][cover[:, 0] == fid][0]
            feat.SetField("woody", w)
            layer.SetFeature(feat)
        feat = None
    data_source = None


# Hardcode the inputs
precip = r'S:\sturt\dingo_fence_dem_ads\woody_veg_mapping\Classification\FinalModel\ADS_strips_classified\QLD_strip_class.tif'
evap = r'S:\sturt\dingo_fence_dem_ads\woody_veg_mapping\Classification\FinalModel\ADS_strips_classified\NSW_north_strip_class.tif'
aridity = 

points = r'S:\sturt\dingo_fence_dem_ads\Topographic_Analysis\analysis\Crest_Points_QLD-NSW_w_metrics.shp'
process_points(points, )

points = r'S:\sturt\dingo_fence_dem_ads\Topographic_Analysis\analysis\Crest_Points_SA-NSW_w_metrics.shp'
process_points(points, )

print("Completed processing")