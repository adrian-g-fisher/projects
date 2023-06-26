#!/usr/bin/env python
"""

Merge images

Set band names

"""

import os
import sys
import argparse
import glob
import numpy as np
from rios import applier


def mergeImages(info, inputs, outputs, otherargs):
    """
    Merges aba and abb Sentinel-2 images.
    """
    blue = inputs.aba[0]
    green = inputs.aba[1]
    red = inputs.aba[2]
    nir = inputs.aba[3]
    re1 = inputs.abb[0]
    re2 = inputs.abb[1]
    re3 = inputs.abb[2]
    nnir = inputs.abb[3]
    swir1 = inputs.abb[4]
    swir2 = inputs.abb[5]
    outputs.mergedImage = np.array([blue, green, red, re1, re2, re3, nir, nnir,
                                    swir1, swir2])
    

def main(abaImage, abbImage, outImage, refImage):
    """
    This sets up RIOS to do the unmixing.
    """
    infiles = applier.FilenameAssociations()
    infiles.aba = abaImage
    infiles.abb = abbImage
    outfiles = applier.FilenameAssociations()
    outfiles.mergedImage = outImage
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(32767)
    controls.setCalcStats(True)
    controls.setOutputDriverName("HFA")
    controls.setReferenceImage(refImage)
    controls.setResampleMethod('bilinear')
    controls.setCreationOptions(["COMPRESSED=YES"])
    controls.setLayerNames(["B2_blue", "B3_green", "B4_red", "B5_RE1", "B6_RE2",
                            "B7_RE3", "B8_NIR", "B8a_NNIR", "B11_SWIR1",
                            "B12_SWIR2"])
    applier.apply(mergeImages, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)


def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Merges aba and abb Sentinel-2 images"))
    p.add_argument("-a", "--abaImage", dest="abaImage", default=None,
                   help=("Input aba Sentinel-2 surface reflectance file."))
    p.add_argument("-b", "--abbImage", dest="abbImage", default=None,
                   help=("Input abb Sentinel-2 surface reflectance file."))
    p.add_argument("-r", "--refImage", dest="refImage", default=None,
                   help=("Input reference Sentinel-2 image."))
    p.add_argument("-o", "--outImage", dest="outImage", default=None,
                   help=("Output merged image."))
    cmdargs = p.parse_args()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    main(cmdargs.abaImage, cmdargs.abbImage, cmdargs.outImage, cmdargs.refImage)
