#!/usr/bin/env python
import os
import sys
import glob
import numpy as np
from rios import applier

def calcStatsTotal(info, inputs, outputs, otherargs):
    """
    This function is called from RIOS to calculate the stats image from the
    input files.
    """

    stack = np.array(inputs.fc_list).astype(np.float32)
    green_stack = stack[:, 1, :, :] - 100
    dead_stack = stack[:, 2, :, :] - 100
    total_stack = green_stack + dead_stack
    
    total_nodata = (total_stack == 0)
    total_stack[total_stack < 0] = 0
    total_stack[total_stack > 100] = 100
    total_stack = np.ma.masked_where(total_nodata == 1, total_stack)
    
    nodata = (np.sum(total_nodata, axis=0) == stack.shape[0])
    
    meanTotal = np.mean(total_stack, axis=0)
    meanTotal[nodata == 1] = 255
    outputs.stats = np.array([meanTotal]).astype(np.float32)
 

def calcStats(info, inputs, outputs, otherargs):
    """
    This function is called from RIOS to calculate the stats image from the
    input files.
    """

    stack = np.array(inputs.fc_list).astype(np.float32)
    bare_stack = stack[:, 0, :, :] - 100
    green_stack = stack[:, 1, :, :] - 100
    dead_stack = stack[:, 2, :, :] - 100
    
    bare_nodata = (stack[:, 0, :, :] == 0)
    bare_stack[bare_stack < 0] = 0
    bare_stack[bare_stack > 100] = 100
    bare_stack = np.ma.masked_where(bare_nodata == 1, bare_stack)
    
    green_nodata = (stack[:, 1, :, :] == 0)
    green_stack[green_stack < 0] = 0
    green_stack[green_stack > 100] = 100
    green_stack = np.ma.masked_where(green_nodata == 1, green_stack)
    
    dead_nodata = (stack[:, 2, :, :] == 0)
    dead_stack[dead_stack < 0] = 0
    dead_stack[dead_stack > 100] = 100
    dead_stack = np.ma.masked_where(dead_nodata == 1, dead_stack)
    
    nodata = (np.sum(green_nodata, axis=0) == stack.shape[0])
    
    if otherargs.stats == 'mean':
        meanBare = np.mean(bare_stack, axis=0)
        meanBare[nodata == 1] = 255
        meanGreen = np.mean(green_stack, axis=0)
        meanGreen[nodata == 1] = 255
        meanDead = np.mean(dead_stack, axis=0)
        meanDead[nodata == 1] = 255
        outputs.stats = np.array([meanBare, meanGreen, meanDead]).astype(np.float32)
    
    if otherargs.stats == 'stdev':
        stdevBare = np.std(bare_stack, axis=0)
        stdevBare[nodata == 1] = 255
        stdevGreen = np.std(green_stack, axis=0)
        stdevGreen[nodata == 1] = 255
        stdevDead = np.std(dead_stack, axis=0)
        stdevDead[nodata == 1] = 255
        outputs.stats = np.array([stdevBare, stdevGreen, stdevDead]).astype(np.float32)

inDir = r'S:\fowlers_gap\imagery\landsat\seasonal_fractional_cover'
dstDir = r'S:\fowlers_gap\imagery\landsat\fractional_cover_timeseries_statistics'

imageList = glob.glob(os.path.join(inDir, r'*_subset.tif'))
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

controls.setLayerNames(['totalcover_mean'])
outfiles.stats = os.path.join(dstDir, r'fowlers_fc_totalcover_mean_198712_202205.tif')
applier.apply(calcStatsTotal, infiles, outfiles, otherArgs=otherargs, controls=controls)

otherargs.stats = 'stdev'
controls.setLayerNames(['Bare_stdev', 'PV_stdev', 'NPV_stdev'])
outfiles.stats = os.path.join(dstDir, r'fowlers_fc_stdev_198712_202205.tif')
applier.apply(calcStats, infiles, outfiles, otherArgs=otherargs, controls=controls)

otherargs.stats = 'mean'
controls.setLayerNames(['Bare_mean', 'PV_mean', 'NPV_mean'])
outfiles.stats = os.path.join(dstDir, r'fowlers_fc_mean_198712_202205.tif')
applier.apply(calcStats, infiles, outfiles, otherArgs=otherargs, controls=controls)