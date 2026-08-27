#!/usr/bin/env python
"""
Calculates three statistic images from the time series of seasonal fracional
cover:
- 36 year (141 seasons) time series (198712-202302)
- 13 year (52 seasons) time series as a sheep grazing property (199609-200911)
- 13 year (52 seasons) time series as a conservation property  (201003-202302)
"""

import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr, osr
from rios import applier, rat


def calcStats(info, inputs, outputs, otherargs):
    """
    This function is called from RIOS to calculate the stats image from the
    input files.
    """
    stack = np.array(inputs.fc_list).astype(np.float32)
    green_stack = stack[:, 1, :, :] - 100
    dead_stack = stack[:, 2, :, :] - 100
    bare_stack = stack[:, 0, :, :] - 100
    
    green_nodata = (stack[:, 1, :, :] == 0)
    green_stack[green_stack < 0] = 0
    green_stack[green_stack > 100] = 100
    green_stack = np.ma.masked_where(green_nodata == 1, green_stack)
    
    dead_nodata = (stack[:, 2, :, :] == 0)
    dead_stack[dead_stack < 0] = 0
    dead_stack[dead_stack > 100] = 100
    dead_stack = np.ma.masked_where(dead_nodata == 1, dead_stack)
    
    bare_nodata = (stack[:, 0, :, :] == 0)
    bare_stack[bare_stack < 0] = 0
    bare_stack[bare_stack > 100] = 100
    bare_stack = np.ma.masked_where(bare_nodata == 1, bare_stack)
    
    nodata = (np.sum(green_nodata, axis=0) == stack.shape[0])
    
    meanGreen = np.mean(green_stack, axis=0)
    meanGreen[nodata == 1] = 255
    stdGreen = np.std(green_stack, axis=0)
    stdGreen[nodata == 1] = 255
    minGreen = np.min(green_stack, axis=0)
    minGreen[nodata == 1] = 255
    maxGreen = np.max(green_stack, axis=0)
    maxGreen[nodata == 1] = 255
    
    meanDead = np.mean(dead_stack, axis=0)
    meanDead[nodata == 1] = 255
    stdDead = np.std(dead_stack, axis=0)
    stdDead[nodata == 1] = 255
    minDead = np.min(dead_stack, axis=0)
    minDead[nodata == 1] = 255
    maxDead = np.max(dead_stack, axis=0)
    maxDead[nodata == 1] = 255
    
    meanBare = np.mean(bare_stack, axis=0)
    meanBare[nodata == 1] = 255
    stdBare = np.std(bare_stack, axis=0)
    stdBare[nodata == 1] = 255
    minBare = np.min(bare_stack, axis=0)
    minBare[nodata == 1] = 255
    maxBare = np.max(bare_stack, axis=0)
    maxBare[nodata == 1] = 255
    
    outputs.stats = np.array([meanGreen, stdGreen, minGreen, maxGreen,
                              meanDead, stdDead, minDead, maxDead,
                              meanBare, stdBare, minBare, maxBare,]).astype(np.float32)


inDir = r'S:\witchelina\seasonal_fractional_cover'
outDir = r'S:\witchelina\timeseries_statistic_images'
for (t1, t2) in [[198712, 202302], [199611, 200911], [201003, 202302]]:
    imageList = np.array(glob.glob(os.path.join(inDir, r'*_subset.tif')))
    startList = np.array([int(os.path.basename(f).split('_')[2][1:7]) for f in imageList])
    endList = np.array([int(os.path.basename(f).split('_')[2][7:13]) for f in imageList])
    imageList = list(imageList[(startList >= t1) & (endList <= t2)])
    infiles = applier.FilenameAssociations()
    infiles.fc_list = imageList
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setWindowXsize(64)
    controls.setWindowYsize(64)
    controls.setStatsIgnore(255)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setLayerNames(['PV_mean', 'PV_stdev', 'PV_min', 'PV_max',
                            'NPV_mean', 'NPV_stdev', 'NPV_min', 'NPV_max',
                            'Bare_mean', 'Bare_stdev', 'Bare_min', 'Bare_max'])
    outfiles.stats = os.path.join(outDir, r'timeseries_stats_%i%i.tif'%(t1, t2))
    applier.apply(calcStats, infiles, outfiles, otherArgs=otherargs, controls=controls)
    print('Created %s'%outfiles.stats)