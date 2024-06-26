#!/usr/bin/env python

import glob
import os, sys
from osgeo import gdal
gdal.UseExceptions()


def stackbands():
    srcDir = r'S:\\aust\\modis_surface_reflectance\\MODIS_Terra_8day_MOD09A1-061_singlebands'
    dstDir = r'S:\\aust\\modis_surface_reflectance\\MODIS_Terra_8day_MOD09A1-061_multibands'
    bands = [r'red', r'nir', r'blue', r'green', r'swir12', r'swir16', r'swir22', r'qc']
    band1_list = glob.glob(os.path.join(srcDir, 'MOD09A1.061_sur_refl_b01_doy*_aid0001.tif'))
    for i, b1 in enumerate(band1_list):
        b2 = b1.replace('_b01_', '_b02_')
        b3 = b1.replace('_b01_', '_b03_')
        b4 = b1.replace('_b01_', '_b04_')
        b5 = b1.replace('_b01_', '_b05_')
        b6 = b1.replace('_b01_', '_b06_')
        b7 = b1.replace('_b01_', '_b07_')
        b8 = b1.replace('_b01_', '_qc_500m_')
        images = [b1, b2, b3, b4, b5, b6, b7, b8]
        dstFile = os.path.join(dstDir, os.path.basename(b1).replace('_b1_', '_ms_'))
        if os.path.exists(dstFile) is False:
            VRT = dstFile.replace('.tif', '.vrt')
            outds = gdal.BuildVRT(VRT, images, separate=True)
            outds = gdal.Translate(dstFile, outds)
            for i in range(len(bands)):
                band = outds.GetRasterBand(i+1)
                band.SetDescription(bands[i])
            os.remove(VRT)
            print("Processed %i or %i"%(i, len(band1_list)))    
        else:
            print("Processed %i or %i"%(i, len(band1_list)))

stackbands()