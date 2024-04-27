#!/usr/bin/env python
"""
"""
import os, sys
import numpy as np
import glob
from rios import applier


inDir = r's:\aust\bom_climate_grids\bom_monthly_rainfall\1second\NetCDF'
outDir = r's:\aust\bom_climate_grids\bom_monthly_rainfall\1second\tif'


def create_fireyear(info, inputs, outputs, otherargs):
    stack = np.array(inputs.monthly).astype(np.float32)
    stack = stack[:, 0, :, :]
    annual = np.sum(stack, axis=0)
    outputs.fireyear = np.array([annual])


# Make monthly geotiffs
for year in [1973, 1974, 2010, 2011, 2020]:
    for inFile in glob.glob(os.path.join(inDir, '*_%i.nc'%year)):
        for band in range(1, 13):
            outFile = os.path.join(outDir, 'agcd_v2_precip_total_r001_%i%02d.tif'%(year, band))
            if os.path.exists(outFile) is False:
                cmd = 'gdal_translate -of GTiff -b %i NETCDF:%s:precip %s'%(band, inFile, outFile)
                os.system(cmd)


# Combine into fire year rainfall geotiffs
imageList = []
for fireyear in [[1973, 1974], [2010, 2011]]:
    for y in fireyear:
        if y == min(fireyear):
            months = range(7, 13)
        else:
            months = range(1, 7)
        for m in months:
            imageList.append(os.path.join(outDir, 'agcd_v2_precip_total_r001_%i%02d.tif'%(y, m)))
    outfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\bom\rainfall_%i%i_fireyear.tif'%(
    if os.path.exists(outfile) is False:    
        fireyear[0], fireyear[1])
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        infiles.monthly = imageList
        outfiles.fireyear = outfile
        controls.setStatsIgnore(0)
        controls.setCalcStats(True)
        controls.setOutputDriverName("GTiff")
        applier.apply(create_fireyear, infiles, outfiles, otherArgs=otherargs, controls=controls)
