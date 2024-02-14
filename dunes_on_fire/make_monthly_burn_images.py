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
    OUT = [outputs.JAN, outputs.FEB, outputs.MAR, outputs.APR,
           outputs.MAY, outputs.JUN, outputs.JUL, outputs.AUG,
           outputs.SEP, outputs.OCT, outputs.NOV, outputs.DEC]
    for m in range(1, 13):
        startdate = datetime.date(year=y, month=m, day=1)
        startday = (datetime.date(y, 1, 1) - startdate).days + 1
        if m < 12:
            enddate = datetime.date(year=y, month=m, day=1) - datetime.timedelta(days=1)
        else:
            enddate = datetime.date(year=y+1, month=1, day=1) - datetime.timedelta(days=1)
        endday = (datetime.date(y, 1, 1) - enddate).days + 1
        monthly = (inputs.annual >= startday) & (inputs.annual <= endday)
        OUT[m-1] = monthly.astype(np.uint8)


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
    outfiles.JAN = os.path.join(outdir, r"burnt_area_%s01.tif"%year)
    outfiles.FEB = os.path.join(outdir, r"burnt_area_%s02.tif"%year)
    outfiles.MAR = os.path.join(outdir, r"burnt_area_%s03.tif"%year)
    outfiles.APR = os.path.join(outdir, r"burnt_area_%s04.tif"%year)
    outfiles.MAY = os.path.join(outdir, r"burnt_area_%s05.tif"%year)
    outfiles.JUN = os.path.join(outdir, r"burnt_area_%s06.tif"%year)
    outfiles.JUL = os.path.join(outdir, r"burnt_area_%s07.tif"%year)
    outfiles.AUG = os.path.join(outdir, r"burnt_area_%s08.tif"%year)
    outfiles.SEP = os.path.join(outdir, r"burnt_area_%s09.tif"%year)
    outfiles.OCT = os.path.join(outdir, r"burnt_area_%s10.tif"%year)
    outfiles.NOV = os.path.join(outdir, r"burnt_area_%s11.tif"%year)
    outfiles.DEC = os.path.join(outdir, r"burnt_area_%s12.tif"%year)
    otherargs.year = int(year)
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    applier.apply(create_monthly, infiles, outfiles, otherArgs=otherargs, controls=controls)
    clrTbl = np.array([[1, 255, 127, 127, 255]])
    for m in [outputs.JAN, outputs.FEB, outputs.MAR, outputs.APR,
              outputs.MAY, outputs.JUN, outputs.JUL, outputs.AUG,
              outputs.SEP, outputs.OCT, outputs.NOV, outputs.DEC]:
        rat.setColorTable(m, clrTbl)


imagedir = r"C:\Users\Adrian\Documents\temp\annual_tiffs"
for imagefile in glob.glob(os.path.join(imagedir, r"*.tif")):
    make_monthly_images(imagefile)