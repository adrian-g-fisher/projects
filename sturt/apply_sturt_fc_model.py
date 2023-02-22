#!/usr/bin/env python
"""

"""

import os
import sys
import argparse
import glob
import numpy as np
from rios import applier


def sort_inputs(x):
    """
    Sorts out the surface reflectance values into what the model expects. 
    """
    n = x.shape[1]
    for i in range(0, n):
        for j in range(i, n):
            ijPred = np.transpose(np.array([x[:, i] * x[:, j]]))
            x = np.hstack((x, ijPred))
    x = x - np.array([0.0785686,  0.1399865,  0.27108628, 0.1399865,
                      0.1399865,  0.1399865,  0.0063463,  0.01131059,
                      0.02175451, 0.01131059, 0.01131059, 0.01131059,
                      0.02018257, 0.03886226, 0.02018257, 0.02018257,
                      0.02018257, 0.07517257, 0.03886226, 0.03886226,
                      0.03886226, 0.02018257, 0.02018257, 0.02018257,
                      0.02018257, 0.02018257, 0.02018257])
    x = x / np.array([0.24520677, 0.45107266, 0.76461905, 0.45107266,
                      0.45107266, 0.45107266, 0.03956141, 0.07067782,
                      0.1218536,  0.07067782, 0.07067782, 0.07067782,
                      0.12756574, 0.22334427, 0.12756574, 0.12756574,
                      0.12756574, 0.4052223,  0.22334427, 0.22334427, 
                      0.22334427, 0.12756574, 0.12756574, 0.12756574,
                      0.12756574, 0.12756574, 0.12756574])
    return(x)

def unmixImage(info, inputs, outputs, otherargs):
    """
    Converts a surface reflectance image into a living/dead/bare image.
    """
    constants = [0.18168738, 0.05579159, 0.7625209]
    coefficients = [[ 1.65616620,  0.00000000, -4.06474161,  0.00000000,
                      0.00000000,  0.00000000,  0.00000000,  0.00000000,
                      0.00000000,  0.00000000,  0.00000000,  0.00000000,
                      0.06194535,  0.00000000,  0.00877961,  0.00041568,
                      0.00001191,  0.00000000,  0.00000000,  0.00000000,
                      0.00000000,  0.00000008,  0.00000001,  0.00000001,
                      0.00000001,  0.00000001,  0.00000001],
                    [-0.72025198,  0.00000000,  0.58281106,  0.00000000,
                      0.00000000,  0.00000000,  0.00000000,  0.00000000,
                      0.00000000,  0.00000000,  0.00000000,  0.00000000,
                     -0.13479316,  0.00000000, -0.01910398, -0.00090450,
                     -0.00002591,  0.00000000,  0.00000000,  0.00000000,
                      0.00000000, -0.00000018, -0.00000003, -0.00000001,
                     -0.00000002, -0.00000002, -0.00000002],
                    [-0.93591690,  0.00000000,  3.48193455,  0.00000000,
                      0.00000000,  0.00000000,  0.00000000,  0.00000000,
                      0.00000000,  0.00000000,  0.00000000,  0.00000000,
                      0.07284793,  0.00000000,  0.01032433,  0.00048882,
                      0.00001400,  0.00000000,  0.00000000,  0.00000000,
                      0.00000000,  0.00000010,  0.00000002,  0.00000001,
                      0.00000001,  0.00000001,  0.00000001]]

    sr = inputs.sr
    inshape = sr.shape

    # Remove zeros
    nodata = (inputs.sr[0] == 32767)
    sr[sr <= 0] = 1
    sr[sr == 32767] = 1
    
    # Reshape to n x b
    sr = np.reshape(sr, (inshape[0], inshape[1]*inshape[2])).transpose()

    # Rescale to floating point from 0 to 1
    sr = sr.astype(np.float32) / 10000.0

    # Add interactive terms
    sr = sort_inputs(sr)
    
    # Unmix and convert to percentages
    living = (np.sum(sr * coefficients[0], axis=1) + constants[0]) * 100
    dead =   (np.sum(sr * coefficients[1], axis=1) + constants[1]) * 100
    bare =   (np.sum(sr * coefficients[2], axis=1) + constants[2]) * 100
    fc = np.vstack([bare, living, dead])

    # Reshape back to an array with 3 bands
    fc = np.reshape(fc, (3, inshape[1], inshape[2]))
    fc[fc < 0] = 0
    fc[fc > 100] = 100
    fc[:, nodata] = 255
    outputs.fc = fc.astype(np.uint8)


def main(inImage, outImage):
    """
    This sets up RIOS to do the unmixing.
    """
    infiles = applier.FilenameAssociations()
    infiles.sr = inImage
    outfiles = applier.FilenameAssociations()
    outfiles.fc = outImage
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(255)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    applier.apply(unmixImage, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)


def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Creates sturt fractional cover images from " +
                         "JRSRP surface reflectance Landsat imagery"))
    p.add_argument("-i", "--inImage", dest="inImage", default=None,
                   help=("Input Landsat surface reflectance file."))
    p.add_argument("-o", "--outImage", dest="outImage", default=None,
                   help=("Output arid fractional cover file."))
    p.add_argument("--inDir", dest="inDir", default=None,
                   help=("Directory with Landsat surface reflectance files."))
    p.add_argument("--outDir", dest="outDir", default=None,
                   help=("Directory for fractional cover files."))
    cmdargs = p.parse_args()
    if cmdargs.inDir is None and cmdargs.outDir is None:
        if cmdargs.inImage is None or cmdargs.outImage is None:
            p.print_help()
            print("Must name in/out images or directories.")
            sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    if cmdargs.inDir is not None:
        for inImage in glob.glob(os.path.join(cmdargs.inDir, '*.tif')):
            outImage = os.path.join(cmdargs.outDir,
                       os.path.basename(inImage).replace('.tif', '_sturt.tif'))
            if os.path.exists(outImage) is False:
                main(inImage, outImage)
    else:
        main(cmdargs.inImage, cmdargs.outImage)
