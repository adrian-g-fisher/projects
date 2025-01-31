#!/usr/bin/env python

"""

Adapted from Landsat water index analysis

"""


import os
import sys
import argparse
import glob
import numpy as np
from rios import applier


#def make_wi_image(info, inputs, outputs, otherargs):
#    """
#    This calculates the water index.
#    """
#    green = inputs.sr[2].astype(np.float32)/10000.0
#    red = inputs.sr[3].astype(np.float32)/10000.0
#    nir = inputs.sr[7].astype(np.float32)/10000.0
#    swir1 = inputs.sr[9].astype(np.float32)/10000.0
#    swir2 = inputs.sr[10].astype(np.float32)/10000.0
#    c = [1.7204, 171, 3, -70, -45, -71]
#    wi = (c[0] + c[1]*green + c[2]*red + c[3]*nir + c[4]*swir1 + c[5]*swir2)
#   outputs.wi = np.array([wi])


def make_wi_image(info, inputs, outputs, otherargs):
    """
    This calculates the water index.
    """
    green = inputs.sr[2].astype(np.float32)/10000.0
    swir2 = inputs.sr[10].astype(np.float32)/10000.0
    wi = (green - swir2) / (green + swir2)
    outputs.wi = np.array([wi])


def main(inImage, outImage):
    """
    """
    infiles = applier.FilenameAssociations()
    infiles.sr = inImage
    outfiles = applier.FilenameAssociations()
    outfiles.wi = outImage
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(32767)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    applier.apply(make_wi_image, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)


def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Creates a water index image from " +
                         "DEA Sentinel-2 NBART imagery"))
    p.add_argument("-i", "--inImage", dest="inImage", default=None,
                   help=("Input Sentinel-2 NBART image (tif)."))
    p.add_argument("-o", "--outImage", dest="outImage", default=None,
                   help=("Output water index image (tif)."))
    cmdargs = p.parse_args()
    if cmdargs.inImage is None or cmdargs.outImage is None:
        p.print_help()
        print("Must name in and out images.")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    main(cmdargs.inImage, cmdargs.outImage)