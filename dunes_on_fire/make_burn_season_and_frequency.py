#!/usr/bin/env python
"""

"""
import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import cuiprogress
import datetime


def create_outputs(info, inputs, outputs, otherargs):
    """
    """
    
    # Make frequency image
    stack = np.array(inputs.images).astype(np.uint8)
    frequency = np.sum(stack[:, 0, :, :], axis=0)
    outputs.frequency = np.array([frequency])
    
    # Make seasonal image
    months = otherargs.months
    summer = np.sum(stack[((months == 12) |
                           (months == 1) |
                           (months == 2)), 0, :, :], axis=0)
    autumn =  np.sum(stack[((months == 3) |
                            (months == 4) |
                            (months == 5)), 0, :, :], axis=0)
    winter =  np.sum(stack[((months == 6) |
                            (months == 7) |
                            (months == 8)), 0, :, :], axis=0)
    spring =  np.sum(stack[((months == 9) |
                            (months == 10) |
                            (months == 11)), 0, :, :], axis=0)
    season = np.zeros_like(summer).astype(np.uint8)
    season[(summer > autumn) & (summer > winter) & (summer > spring)] = 1
    season[(autumn > summer) & (autumn > winter) & (autumn > spring)] = 2
    season[(winter > summer) & (winter > autumn) & (winter > spring)] = 3
    season[(spring > summer) & (spring > autumn) & (spring > winter)] = 4
    outputs.season = np.array([season])


def make_season_and_frequency(imageList, monthsList):
    """
    This sets up RIOS
    """
    outdir = r"S:\aust\modis_burned_area"
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.images = imageList
    otherargs.months = monthsList
    outfiles.season = os.path.join(outdir, r"burnt_area_season_200011-202313.tif")
    outfiles.frequency = os.path.join(outdir, r"burnt_area_frequency_200011-202312.tif")
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar())
    applier.apply(create_outputs, infiles, outfiles, otherArgs=otherargs, controls=controls)


imagedir = r"S:\aust\modis_burned_area\monthly_tiffs"
imageList = glob.glob(os.path.join(imagedir, r"*.tif"))
monthsList = np.array([int(x[-6:-4]) for x in imageList])
make_season_and_frequency(imageList, monthsList)