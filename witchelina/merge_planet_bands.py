#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
from rios import applier
from osgeo import gdal, gdalconst


def mergeBandsNow(info, i, outputs, otherargs):
    outputs.m = np.array([i.b1[0], i.b2[0], i.b3[0], i.b4[0], i.b5[0]])


def mergeBands(projName, inDir, outDir):
    dstFile = os.path.join(outDir, r'%s_multiband.tif'%projName)
    if os.path.exists(dstFile) is False:
        infiles = applier.FilenameAssociations()
        infiles.b1 = os.path.join(inDir, r'%s_%s.tif'%(projName, r'band1'))
        infiles.b2 = os.path.join(inDir, r'%s_%s.tif'%(projName, r'band2'))
        infiles.b3 = os.path.join(inDir, r'%s_%s.tif'%(projName, r'band3'))
        infiles.b4 = os.path.join(inDir, r'%s_%s.tif'%(projName, r'band4'))
        infiles.b5 = os.path.join(inDir, r'%s_%s.tif'%(projName, r'band5'))
        outfiles = applier.FilenameAssociations()
        outfiles.m = dstFile
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        controls.setCalcStats(True)
        controls.setOutputDriverName("GTiff")
        applier.apply(mergeBandsNow, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
        # Add GCP info to output file
        source_ds = gdal.Open(infiles.b1, gdalconst.GA_ReadOnly)
        gcpcount = source_ds.GetGCPCount( )
        gcp = source_ds.GetGCPs()
        gcpproj = source_ds.GetGCPProjection()

        ds = gdal.Open(outfiles.m, gdalconst.GA_Update)
        ds.SetGCPs(gcp, gcpproj)
        ds = None
    
    print("Processing completed for %s"%os.path.basename(dstFile))


# 2013 RapidEye images
inDir = r'S:\witchelina\planet\2013\bands'
outDir = r'S:\witchelina\planet\2013\images'
projNames = glob.glob(os.path.join(inDir, '*_band1.tif'))
projNames = [os.path.basename(b).replace('_band1.tif', '') for b in projNames]
for projectName in projNames:
    mergeBands(projectName, inDir, outDir)