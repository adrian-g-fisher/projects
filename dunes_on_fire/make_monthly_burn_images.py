#!/usr/bin/env python
"""

"""
import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import rat
import datetime


def create_monthly(info, inputs, outputs, otherargs):
    """
    Splits the annual image into monthly images using day of year.
    """
    y = otherargs.year
    m = otherargs.month
    startdate = datetime.date(year=y, month=m, day=1)
    startday = (startdate - datetime.date(y, 1, 1)).days + 1
    if m < 12:
        enddate = datetime.date(year=y, month=m+1, day=1) - datetime.timedelta(days=1)
    else:
        enddate = datetime.date(year=y+1, month=1, day=1) - datetime.timedelta(days=1)
    endday = (enddate - datetime.date(y, 1, 1)).days + 1
    monthly = (inputs.annual >= startday) & (inputs.annual <= endday)
    outputs.monthly = monthly.astype(np.uint8)


def make_monthly_images(imagefile):
    """
    This sets up RIOS to create monthly images from an annual image.
    """
    outdir = r"C:\Users\Adrian\Documents\temp\monthly_tiffs"
    year = imagefile.replace(r".tif", "").split(r"_")[-1]
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.annual = imagefile
    for m in range(1, 13):
        outfiles.monthly = os.path.join(outdir, r"burnt_area_%s%02d.tif"%(year, m))
        otherargs.year = int(year)
        otherargs.month = m
        controls.setStatsIgnore(0)
        controls.setCalcStats(True)
        controls.setOutputDriverName("GTiff")
        if os.path.exists(outfiles.monthly) is False:
            applier.apply(create_monthly, infiles, outfiles, otherArgs=otherargs, controls=controls)
            clrTbl = np.array([[1, 255, 127, 127, 255]])
            rat.setColorTable(outfiles.monthly, clrTbl)


imagedir = r"C:\Users\Adrian\Documents\temp\annual_tiffs"
for imagefile in glob.glob(os.path.join(imagedir, r"*.tif")):
    make_monthly_images(imagefile)
    print(os.path.basename(imagefile))
    