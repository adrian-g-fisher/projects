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


def make_frequency_image(imageList, outfile):
    """
    This sets up RIOS
    """

    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.images = imageList
    outfiles.probability = outfile
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

# Sampling interval 2002-2020
outdir = r"S:\hay_plain\pw_absence_model"
outfile = os.path.join(outdir, r"pw_absence_probability_2002-2020.tif")
startDate = datetime.date(year=2002, month=9, day=1)
endDate = datetime.date(year=2020, month=6, day=1)
#make_frequency_image(imageList[(dateList >= startDate) & (dateList <= endDate)].tolist(), outfile)

# 2000-2005
outfile = os.path.join(outdir, r"pw_absence_probability_2000-2005.tif")
startDate = datetime.date(year=1999, month=12, day=1)
endDate = datetime.date(year=2004, month=9, day=1)
make_frequency_image(imageList[(dateList >= startDate) & (dateList <= endDate)].tolist(), outfile)

# 2005-2010
outfile = os.path.join(outdir, r"pw_absence_probability_2005-2010.tif")
startDate = datetime.date(year=2004, month=12, day=1)
endDate = datetime.date(year=2009, month=9, day=1)
make_frequency_image(imageList[(dateList >= startDate) & (dateList <= endDate)].tolist(), outfile)

# 2010-2015
outfile = os.path.join(outdir, r"pw_absence_probability_2010-2015.tif")
startDate = datetime.date(year=2009, month=12, day=1)
endDate = datetime.date(year=2014, month=9, day=1)
make_frequency_image(imageList[(dateList >= startDate) & (dateList <= endDate)].tolist(), outfile)

# 2015-2020
outfile = os.path.join(outdir, r"pw_absence_probability_2015-2020.tif")
startDate = datetime.date(year=2014, month=12, day=1)
endDate = datetime.date(year=2019, month=9, day=1)
make_frequency_image(imageList[(dateList >= startDate) & (dateList <= endDate)].tolist(), outfile)