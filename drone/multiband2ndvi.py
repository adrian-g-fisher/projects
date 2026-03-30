#!/usr/bin/env python
"""
This makes a multiband image from single band images.
"""

import os
import sys
import argparse
import glob
import numpy as np
from osgeo import gdal
from rios import applier
gdal.UseExceptions()


def makeDroneNDVI(info, inputs, outputs, otherargs):
    """
    """
    nodata = info.getNoDataValueFor(inputs.drone)
    red = inputs.drone[2]
    nir = inputs.drone[4]
    sumRedNir = np.where(red + nir == 0, 1, red + nir)
    ndvi = (nir - red) / sumRedNir
    ndvi[red + nir == 0] = 2
    ndvi[(red == nodata) | (nir == nodata)] = 2
    outputs.ndvi = np.array([ndvi]).astype(np.float32)


def makeNDVI(inDir, outDir):
    """
    """
    for inimage in glob.glob(os.path.join(inDir, "*_multiband.tif")):
        outimage = os.path.join(outDir, os.path.basename(inimage).replace("_multiband.tif", "_ndvi.tif"))
        infiles = applier.FilenameAssociations()
        infiles.drone = inimage
        outfiles = applier.FilenameAssociations()
        outfiles.ndvi = outimage
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        controls.setStatsIgnore(2)
        controls.setCalcStats(True)
        controls.setOutputDriverName("GTiff")
        applier.apply(makeDroneNDVI, infiles, outfiles, otherArgs=otherargs, controls=controls)


def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Makes an NDVI image from a multiband image."))
    p.add_argument("-i", "--inDir", dest="inDir", default=None,
                   help=("Directory containing input images"))
    p.add_argument("-o", "--outDir", dest="outDir", default=None,
                   help=("Directory for output images"))  
    cmdargs = p.parse_args()
    if (cmdargs.inDir is None or cmdargs.outDir is None):
        p.print_help()
        print("Must give in and out directory names.")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    makeNDVI(cmdargs.inDir, cmdargs.outDir)