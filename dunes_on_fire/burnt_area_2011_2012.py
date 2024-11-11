#!/usr/bin/env python
"""

Create burnt area raster for 201107-201206

Then vectorise, merge polygons, and calculate area

"""
import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import cuiprogress
import datetime


def create_annual(info, inputs, outputs, otherargs):
    """
    """
    stack = np.array(inputs.images).astype(np.uint8)
    annual = np.sum(stack[:, 0, :, :], axis=0)
    annual[annual > 0] = 1
    outputs.annual = np.array([annual])


def make_annual_burnt(imageList):
    """
    This sets up RIOS
    """
    outdir = r"S:\aust\modis_burned_area"
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.images = imageList
    outfiles.annual = os.path.join(outdir, r"burnt_area_201107-201206.tif")
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar())
    applier.apply(create_annual, infiles, outfiles, otherArgs=otherargs, controls=controls)

dates = [201107, 201108, 201109, 201110, 201111, 201112,
         201201, 201202, 201203, 201204, 201205, 201206]
imagedir = r"S:\aust\modis_burned_area\monthly_tiffs"
imageList = np.array(glob.glob(os.path.join(imagedir, r"*.tif")))
dateList = np.array([int(x[-10:-4]) for x in imageList])
subset = []
for d in dates:
    subset.append(imageList[dateList == d][0])

make_annual_burnt(subset)