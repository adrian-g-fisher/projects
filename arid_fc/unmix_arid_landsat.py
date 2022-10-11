#!/usr/bin/env python
"""
Background

This creates arid zone fractional cover images from JRSRP surface reflectance
Landsat images. It will work on TM, ETM+, or seasonal composites, but OLI
imagery will need correcting first (Flood, 2014).

It follows the method as described in Shumack et al (2021). Most of the code was
developed by Sam Shumack, with minor modification by Adrian Fisher. It needs
AZN_M.csv, which contains the endmember data produced by Sam. The output is in
GeoTiff format.

This needs the fast-nnls package to be installed from GitHub with the following:
    pip install git+https://github.com/lostsea/fast-nnls.git

References

Flood, N., 2014. Continuity of reflectance data between Landsat-7 ETM+ and
    Landsat-8 OLI, for both top-of-atmosphere and surface reflectance: a study
    in the Australian landscape. Remote Sensing, 6, 7952–7970.
    https://doi.org/10.3390/rs6097952

Shumack, S., Fisher, A., Hesse, P.P., (2021), Refining medium resolution
    fractional cover for arid Australia to detect vegetation dynamics and wind
    erosion susceptibility on longitudinal dunes. Remote Sensing of Environment,
    265, https://doi.org/10.1016/j.rse.2021.112647

Examples

The script can be run at the command line with the following arguments

  -h, --help            show this help message and exit
  -i INIMAGE, --inImage INIMAGE
                        Input Landsat surface reflectance file.
  -o OUTIMAGE, --outImage OUTIMAGE
                        Output arid fractional cover file.
  --inDir INDIR         Directory with Landsat surface reflectance files.
  --outDir OUTDIR       Directory for arid fractional cover files.

Example 1. Convert one surface reflectance image to arid zone fractional cover

    > python unmix_arid_landsat.py -i surf_ref.tif -o arid_fc.tif

Example 2. Convert many surface reflectance images to arid zone fractional cover

    > python unmix_arid_landsat.py --inDir C:\surf_ref --outDir C:\arid_fc

"""


import os
import sys
import argparse
import itertools
import glob
import numpy as np
from fastnnls import fnnls
from rios import applier
from rios import cuiprogress
from csv import reader


def add_interactive_terms(X):
    """
    This adds the interactive terms as used by Shumack et al (2021) (similar to 
    Scarth et al. (2010) and Guerschman et al. (2015) but fewer interactive
    terms are used here).
    """
    combos = list(itertools.combinations(range(0, X.shape[1]), 2))
    
    # bands 1-6 = reflectance
    
    # bands 7-12 = log transforms
    for b in range(0, 6):
        logb = np.log(X[:, b])
        X = np.hstack((X, logb.reshape((logb.shape[0], 1))))
    
    # bands 13-27 = product of each log band combination
    for c in combos:
        logbprod = X[:, c[0] + 6] * X[:, c[1] + 6]
        X = np.hstack((X, logbprod.reshape((logbprod.shape[0], 1))))
    
    # bands 28-42 = normalised band ratios
    for c in combos:
        norm_ratio = (X[:, c[0]] - X[:, c[1]]) / (X[:, c[0]] + X[:, c[1]])
        X = np.hstack((X, norm_ratio.reshape((norm_ratio.shape[0], 1))))
    
    return X



def unmix(X, M, w=0.2, c=3):
    """
    Unmixes an image into fractional cover.
    X is an array with n pixels times b bands (including all interactive terms).
    M is an array with c cover types times b bands.
    c = 3 for bare, PV and NPV
    w = 0.2 is the weighting factor for the sum-to-one constraint.
    """
    
    # Add ones for sum to one constraint
    Mw = np.hstack((M, (np.ones((M.shape[0], 1))) * w))
    X = np.hstack((X, (np.ones((X.shape[0], 1))) * w))
    
    # Create array to store fractional cover percentage values
    y_pred = np.zeros((X.shape[0], c), dtype=np.float32)
    
    # Iterate through the pixels
    for i, x in enumerate(X):
        solution = fnnls(Mw @ Mw.T, Mw @ x)
        y_pred[i] = solution
    
    return y_pred


def unmixImage(info, inputs, outputs, otherargs):
    """
    Called from rios to do the unmixing.
    """
    bands = inputs.sr.shape[0]
    if bands == 6:
        sr = inputs.sr
    elif bands > 6:
        # Select only the blue, green, red, nir, swir1 and swir2 bands
        sr = inputs.sr[1:7]
    else:
        print("Error: number of bands does not match the Landsat endmembers")
        sys.exit()
    
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
    sr = add_interactive_terms(sr)
    
    # Unmix and convert to percentages
    fc = unmix(sr, otherargs.M) * 100
    
    # Reshape back to an array with 3 bands
    fc = np.reshape(fc.transpose(), (3, inshape[1], inshape[2]))
    fc[:, nodata] = 255
    
    outputs.fc = fc.astype(np.uint8)


def main(inImage, outImage):
    """
    This sets up RIOS to do the unmixing.
    """
    # Retrieve endmembers from csv file
    endmembers = 'AZN_M.csv'
    M = np.loadtxt(endmembers, delimiter=',')
    
    # Set up RIOS
    infiles = applier.FilenameAssociations()
    infiles.sr = inImage
    outfiles = applier.FilenameAssociations()
    outfiles.fc = outImage
    otherargs = applier.OtherInputs()
    otherargs.M = M
    controls = applier.ApplierControls()
    controls.setStatsIgnore(255)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar())
    applier.apply(unmixImage, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)
    
    
def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Creates arid zone fractional cover images from " +
                         "JRSRP surface reflectance Landsat imagery"))
    p.add_argument("-i", "--inImage", dest="inImage", default=None,
                   help=("Input Landsat surface reflectance file."))
    p.add_argument("-o", "--outImage", dest="outImage", default=None,
                   help=("Output arid fractional cover file."))
    p.add_argument("--inDir", dest="inDir", default=None,
                   help=("Directory with Landsat surface reflectance files."))
    p.add_argument("--outDir", dest="outDir", default=None,
                   help=("Directory for arid fractional cover files."))
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
                       os.path.basename(inImage).replace('.tif', '_AZN.tif'))
            if os.path.exists(outImage) is False:
                main(inImage, outImage)
    else:
        main(cmdargs.inImage, cmdargs.outImage)
