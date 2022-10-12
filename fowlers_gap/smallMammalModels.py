#!/usr/bin/env python
import os
import sys
import glob
import datetime
import numpy as np
from rios import applier


def calcLags(info, inputs, outputs, otherargs):
    """
    This function is called from RIOS to calculate the lag image.
    Remember that PV is layer 2 in input FC images.
    """
    # Caculate monthly mean values of PV
    stack = np.array(inputs.imageList).astype(np.float32)
    pvStack = stack[:, 1, :, :] - 100
    sampleYearMonth = otherargs.sampleYearMonth
    dateArray = otherargs.dateArray
    years = np.array([x.year for x in dateArray])
    months = np.array([x.month for x in dateArray])
    yearMonths = []
    pvs = []
    for y in np.unique(years):
        for m in np.unique(months):
            i = np.where((years == y) & (months == m))[0]
            if i.size == 1:
                yearMonths.append(datetime.date(y, m, 1))
                pv = np.squeeze(pvStack[i, :, :])
                pvs.append(pv)
            if i.size > 1:
                yearMonths.append(datetime.date(y, m, 1))
                pv = np.mean(pvStack[i, :, :], axis=0)
                pvs.append(pv)
    yearMonths = np.array(yearMonths)
    pvs = np.array(pvs)
    
    # Now calculate mean values for lag times
    lagArrays = np.zeros((4, pvStack.shape[1], pvStack.shape[2]), dtype=np.float32)
    lagList = [[1, 3], [4, 6], [7, 9], [10, 12]]
    for i in range(4):
        minMonths = lagList[i][0]
        maxMonths = lagList[i][1]
        minDays = ((minMonths - 1) * 30) + 15
        maxDays = ((maxMonths - 1) * 30) + 15
        start = sampleYearMonth - datetime.timedelta(days=minDays)
        start = datetime.date(start.year, start.month, 1)
        end = sampleYearMonth - datetime.timedelta(days=maxDays)
        end = datetime.date(end.year, end.month, 1)
        ind = np.where((yearMonths <= start) & (yearMonths >= end))[0]
        if ind.size == 1:
            lagArrays[i, :, :] = pvs[ind]
        if ind.size > 1:
            lagArrays[i, :, :] = np.mean(pvs[ind], axis=0)
        
    outputs.lagImage = lagArrays


def make_lag_image(imageDate, imageDir):
    """
    Creates an image with the followig bands:
    - Mean PV of images 1-3 month prior
    - Mean PV of images 4-6 month prior
    - Mean PV of images 7-9 month prior
    - Mean PV of images 10-12 month prior
    """
    sampleYearMonth = datetime.date(year=int(imageDate[0:4]), month=int(imageDate[4:6]), day=1) 
    imageList = glob.glob(os.path.join(imageDir, '*.img'))
    dateList = [os.path.basename(x).split('_')[2] for x in imageList]
    dateArray = np.array([datetime.date(year=int(x[0:4]), month=int(x[4:6]), day=int(x[6:8])) for x in dateList])
    infiles = applier.FilenameAssociations()
    infiles.imageList = imageList
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    otherargs.sampleYearMonth = sampleYearMonth
    otherargs.dateArray = dateArray
    controls = applier.ApplierControls()
    controls.setWindowXsize(64)
    controls.setWindowYsize(64)
    controls.setStatsIgnore(255)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setLayerNames(['1-3 PV', '4-6 PV', '7-9 PV', '10-12 PV'])
    outfiles.lagImage = os.path.join(imageDir, r'lag_image_%s.tif'%imageDate)
    applier.apply(calcLags, infiles, outfiles, otherArgs=otherargs, controls=controls)


def applyModels(info, inputs, outputs, otherargs):
    """
    This function is called from RIOS to calculate the small mammal probability
    images.
    """
    # Listed like in the table in the chapter
    c_forresti = [[-2.36190, 0.00000, 0.26522, -0.09280, -0.01435, -0.34504,  0.00000], # Emu
                  [-2.36190, 0.00000, 0.26522, -0.09280, -0.01435, -0.34504, -0.70613], # Unfenced
                  [-2.36190, 0.00000, 0.26522, -0.09280, -0.01435,  0.00000,  0.00000], # Conservation
                  [-2.36190, 0.00000, 0.26522, -0.09280, -0.01435,  0.00000, -0.70613], # Unfenced
                  [-2.36190, 0.00000, 0.26522, -0.09280, -0.01435,  1.36025,  0.00000], # Warrens
                  [-2.36190, 0.00000, 0.26522, -0.09280, -0.01435,  1.36025, -0.70613]] # Unfenced
    c_musculus =  [-1.3530,  0.1042,  0.1105,   0.0000,   0.0000] # No effct of paddock or treatment
    pv_1_3 = inputs.lagImage[0]
    pv_4_6 = inputs.lagImage[1]
    pv_7_9 = inputs.lagImage[2]
    pv_10_12 = inputs.lagImage[3]
    
    # Make arrays for output
    f = np.zeros_like(pv_1_3)
    m = np.zeros_like(pv_1_3)
    
    s = inputs.sites[0]
    if np.max(s) > 0:
        for i in np.unique(s):
            
            # forresti model
            c = c_forresti[i - 1]
            f[s == i] = (c[0] + c[1]*pv_1_3[s == i] +
                                c[2]*pv_4_6[s == i] +
                                c[3]*pv_7_9[s == i] +
                                c[4]*pv_10_12[s == i] +
                                c[5] + c[6])
        
        # musculus model
        c = c_musculus
        m[s > 0] = (c[0] + c[1]*pv_1_3[s > 0] +
                           c[2]*pv_4_6[s > 0] +
                           c[3]*pv_7_9[s > 0] +
                           c[4]*pv_10_12[s > 0])
    
    f[s == 0] = 255
    m[s == 0] = 255
    f[f < 0] = 0
    m[m < 0] = 0
    outputs.forresti = np.array([f])
    outputs.musculus = np.array([m])


def smallMammalModel(imageDate, imageDir):
    """
    Reads in the lag image and the shapefile and applies the different models to
    the pixels from different areas to produce a raster of modelled small mammal
    probability.
    """
    infiles = applier.FilenameAssociations()
    infiles.sites = r'C:\Users\Adrian\OneDrive - UNSW\Documents\student_projects\mphil\matt_smith\data\Fowl_treat_polys_albers.shp'
    infiles.lagImage = os.path.join(imageDir, r'lag_image_%s.tif'%imageDate)
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setWindowXsize(64)
    controls.setWindowYsize(64)
    controls.setStatsIgnore(255)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setBurnAttribute("Id")
    outfiles.forresti = os.path.join(imageDir, r'forresti_image_%s.tif'%imageDate)
    outfiles.musculus = os.path.join(imageDir, r'musculus_image_%s.tif'%imageDate)
    applier.apply(applyModels, infiles, outfiles, otherArgs=otherargs, controls=controls)


imageDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\student_projects\mphil\matt_smith\data\20190414'
imageDate = '20190414'
make_lag_image(imageDate, imageDir)
smallMammalModel(imageDate, imageDir)

imageDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\student_projects\mphil\matt_smith\data\20210419'
imageDate = '20210419'
make_lag_image(imageDate, imageDir)
smallMammalModel(imageDate, imageDir)