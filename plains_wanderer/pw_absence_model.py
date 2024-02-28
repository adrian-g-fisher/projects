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
    NPV is band 2 of [0, 1, 2]
    """
    stack = np.array(inputs.images).astype(np.uint8)
    stack = np.where(stack >= 100, stack - 100, 0)
    count = np.sum((stack[:, 2, :, :] < 41.16), axis=0)
    probability = 100 * (count / otherargs.num_images)
    outputs.probability = np.array([probability]).astype(np.float32)


def make_frequency_image(imageList):
    """
    This sets up RIOS
    """
    outdir = r"S:\hay_plain\pw_absence_model"
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.images = imageList
    outfiles.probability = os.path.join(outdir, r"pw_absence_probability.tif")
    otherargs.num_images = float(len(imageList))
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar())
    applier.apply(create_outputs, infiles, outfiles, otherArgs=otherargs, controls=controls)


# Get the FC images
imagedir = r"S:\hay_plain\landsat\landsat_seasonal_fractional_cover"
imageList = np.array(glob.glob(os.path.join(imagedir, r"*.tif")))
dateList = []
for image in imageList:
    y = int(os.path.basename(image).split("_")[2][1:5])
    m = int(os.path.basename(image).split("_")[2][5:7])
    dateList.append(datetime.date(year=y, month=m, day=1))
dateList = np.array(dateList)

# Seasons start on 3, 6, 9, and 12 months
startDate = datetime.date(year=2002, month=9, day=1)
endDate = datetime.date(year=2020, month=6, day=1)
imageList = imageList[(dateList >= startDate) & (dateList <= endDate)].tolist()
make_frequency_image(imageList)