#!/usr/bin/env python
"""

Difference images for PV and NPV between 201803201805 and 202203202205

"""


import os
import sys
import numpy as np
from rios import applier


def makeDiffImages(info, inputs, outputs, otherargs):
    """
    Makes the difference images.
    """
    landforms = inputs.landforms[0]
    pv2018 = inputs.fc2018[1].astype(np.float32)
    pv2022 = inputs.fc2022[1].astype(np.float32)
    pv_diff = pv2022 - pv2018
    pv_diff[pv2018 == 255] = 32767
    pv_diff[pv2022 == 255] = 32767
    pv_diff[landforms == 1] = 32767
    outputs.pv_diff = np.array([pv_diff])
    
    npv2018 = inputs.fc2018[2].astype(np.float32)
    npv2022 = inputs.fc2022[2].astype(np.float32)
    npv_diff = npv2022 - npv2018
    npv_diff[npv2018 == 255] = 32767
    npv_diff[npv2022 == 255] = 32767
    npv_diff[landforms == 1] = 32767
    outputs.npv_diff = np.array([npv_diff])
    

def calc_diff(fc2018, fc2022):
    """
    Sets up RIOS.
    """
    # Set up RIOS to do the calculation
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    infiles.fc2018 = fc2018
    infiles.fc2022 = fc2022
    infiles.landforms = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\published\2021_01_remote_sensing_of_trophic_cascades\Remote_sensing_of_trophic_cascades_data\timeseries_classes_new.img'
    outfiles.pv_diff = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\living_diff_2018_2022.tif'
    outfiles.npv_diff = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\dead_diff_2018_2022.tif'
    controls.setStatsIgnore(32767)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    applier.apply(makeDiffImages, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)


# Hardcode the inputs
fc2018 = r'S:\sturt\landsat\landsat_seasonal_sturt_model\l8olre_aus_m201803201805_dbia2_subset_sturtmodel.tif'
fc2022 = r'S:\sturt\landsat\landsat_seasonal_sturt_model\lzolre_aus_m202203202205_dbia2_subset_sturtmodel.tif'
calc_diff(fc2018, fc2022)
