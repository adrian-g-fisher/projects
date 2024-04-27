#!/usr/bin/env python
"""
"""
import os, sys, glob
import numpy as np
from osgeo import ogr
from rios import applier
from rios import rat


def simplify_nvis(info, inputs, outputs, otherargs):
    """
    Counts pixels in polygons.
    """
    nvis = inputs.nvis[0]
    simple = np.zeros_like(nvis)
    hummock = [33]    
    woodland_with_hummock = [10, 18, 23, 51, 52, 72]
    mallee_with_hummock = [27, 66]
    pixelLists = [hummock, woodland_with_hummock, mallee_with_hummock]
    for x, p in enumerate(pixelLists):
        for i in p:
            simple[nvis == i] = x + 1
    outputs.simple = np.array([simple]).astype(np.uint8)


infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()
otherargs = applier.OtherInputs()
controls = applier.ApplierControls()
controls.setStatsIgnore(0)
controls.setCalcStats(True)
controls.setOutputDriverName("GTiff")
infiles.nvis = r'C:\Users\Adrian\Documents\NVIS\aus6_0e_mvs.tif'
outfiles.simple = r'C:\Users\Adrian\Documents\NVIS\nvis_simplified.tif'
applier.apply(simplify_nvis, infiles, outfiles, otherArgs=otherargs, controls=controls)
clrTbl = np.array([[1, 222, 193, 124, 255], # Hummock grassland
                   [2, 126, 204, 192, 255], # Woodland with hummock grass
                   [3,   1, 133, 113, 255]])# Mallee with hummock grass
rat.setColorTable(outfiles.simple, clrTbl)