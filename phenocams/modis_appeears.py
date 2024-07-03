#!/usr/bin/env python
"""
From the MOD09 (vrsion 6.1) User Guide:

Bits 0-1: Cloud state
0: Clear
1: Cloudy
2: Mixed
3: Not set, assumed clear

Bit 2: Cloud shadow
0: No
1: Yes

Bits 3-5: Land/water flag
0: Shallow ocean
1: Land
2: Ocean coastlines and lake shorelines
3: Shallow inland water
4: Ephemeral water
5: Deep inland water
6: Continental/moderate ocean
7: Deep ocean

Bits 6-7: Aerosol quantity
0: Climatology
1: Low
2: Average
3: High

Bits 8-9: Cirrus detected
0: None
1: Small
2: Average
3: High

Bit 10: Internal cloud algorithm flag
0: No cloud
1: Cloud

Bit 11: Internal fire algorithm flag
0: No fire
1: Fire

Bit 12: MOD35 snow/ice flag
0: No
1: Yes

Bit 13: Pixel is adjacent to cloud
0: No
1: Yes

Bit 14: Salt pan
0: No
1: Yes

Bit 15: Internal snow
0: No snow
1: Snow
"""

import os, sys
import glob
import unpackqa
import datetime
import hdstats
import numpy as np
from osgeo import gdal
from rios import applier
gdal.UseExceptions()


MOD09A1_info = {'flag_info':{'cloud':[0, 1],         # 0 - CLOUD (1)
                             'cloud_shadow':[2],     # 1 - CLOUD_SHADOW (1)
                             'land_water':[3, 4, 5],
                             'aerosol':[6, 7],
                             'cirrus':[8, 9],
                             'internal_cloud':[10],  # 5 - CLOUD (1)
                             'internal_fire':[11],
                             'mod35_snow_ice':[12],
                             'adjacenct_to_cloud':[13],
                             'salt_pan':[14],
                             'internal_snow':[15]},
                'max_value' : 4294967295,
                'num_bits'  : 32}


def stackbands():
    srcDir = r'S:\\aust\\modis_surface_reflectance\\MODIS_Terra_8day_MOD09A1-061_singlebands'
    dstDir = r'S:\\aust\\modis_surface_reflectance\\MODIS_Terra_8day_MOD09A1-061_multibands'
    bands = [r'red', r'nir', r'blue', r'green', r'swir12', r'swir16', r'swir22']
    band1_list = glob.glob(os.path.join(srcDir, 'MOD09A1.061_sur_refl_b01_doy*_aid0001.tif'))
    for i, b1 in enumerate(band1_list):
        b2 = b1.replace('_b01_', '_b02_')
        b3 = b1.replace('_b01_', '_b03_')
        b4 = b1.replace('_b01_', '_b04_')
        b5 = b1.replace('_b01_', '_b05_')
        b6 = b1.replace('_b01_', '_b06_')
        b7 = b1.replace('_b01_', '_b07_')
        images = [b1, b2, b3, b4, b5, b6, b7]
        dstFile = os.path.join(dstDir, os.path.basename(b1).replace('_b01_', '_'))
        if os.path.exists(dstFile) is False:
            VRT = dstFile.replace('.tif', '.vrt')
            outds = gdal.BuildVRT(VRT, images, separate=True)
            outds = gdal.Translate(dstFile, outds)
            for b in range(len(bands)):
                band = outds.GetRasterBand(b+1)
                band.SetDescription(bands[b])
            os.remove(VRT)
            print("Processed %i of %i"%(i+1, len(band1_list)))    
        else:
            print("Processed %i of %i"%(i+1, len(band1_list)))


def mask_cloud_and_make_medoid(info, inputs, outputs, otherargs):
    nullValue = -28672
    numImages = len(inputs.sr)
    sr = np.array(inputs.sr).astype(np.float32)
    for i in range(numImages):
        sr[i][sr[i] == nullValue] = np.nan
        qa = unpackqa.unpack_to_array(inputs.qc[i], product=MOD09A1_info)[0]
        mask = np.zeros_like(sr[i][0]).astype(np.uint8)
        mask[qa[:, :, 0] == 1] = 1 # cloud
        mask[qa[:, :, 5] == 1] = 1 # cloud    
        mask[qa[:, :, 1] == 1] = 1 # cloud shadow
        sr[i][:, mask == 1] = np.nan
    
    # Now calculate geomedian of inputs.sr as gm
    imgStack = sr.transpose()
    gm = hdstats.nangeomedian_pcm(imgStack, num_threads=4).transpose()
    gm[np.isnan(gm)] = 32767
    outputs.gm = gm.astype(np.uint16)
    

def monthly_mosaics():
    imageDir = r'S:\aust\modis_surface_reflectance\MODIS_Terra_8day_MOD09A1-061_multibands'
    qcDir = r'S:\aust\modis_surface_reflectance\MODIS_Terra_8day_MOD09A1-061_state'
    outDir = r'S:\aust\modis_surface_reflectance\modis_monthly_surface_reflectance'
    imageList = glob.glob(os.path.join(imageDir, '*.tif'))
    doyList = [os.path.basename(i).split('_')[3][3:10] for i in imageList]
    yearList = [datetime.datetime.strptime(doy, "%Y%j").year for doy in doyList]
    monthList = [datetime.datetime.strptime(doy, "%Y%j").month for doy in doyList]
    imageList = np.array(imageList)
    yearList = np.array(yearList)
    monthList = np.array(monthList)
    for year in range(2000, 2025):
        for month in range(1, 13):
            srImages = imageList[(yearList == year) & (monthList == month)]
            qcImages = [os.path.join(qcDir, os.path.basename(srImage).replace('_refl_', '_refl_state_500m_')) for srImage in srImages]
            if len(srImages) > 0:
                outImage = os.path.join(outDir, 'mod09a1061_sr_500m_%i%02d.tif'%(year, month))
                if os.path.exists(outImage) is False:
                    infiles = applier.FilenameAssociations()
                    outfiles = applier.FilenameAssociations()
                    otherargs = applier.OtherInputs()
                    controls = applier.ApplierControls()
                    controls.setStatsIgnore(32767)
                    controls.setCalcStats(True)
                    controls.setOutputDriverName("GTiff")
                    controls.setLayerNames([r'red', r'nir', r'blue', r'green', r'swir12', r'swir16', r'swir22'])
                    infiles.sr = list(srImages)
                    infiles.qc = list(qcImages)
                    outfiles.gm = outImage
                    applier.apply(mask_cloud_and_make_medoid, infiles, outfiles, otherArgs=otherargs, controls=controls)
                    print("Created %s"%os.path.basename(outImage))
                
                if os.path.basename(outImage) == 'mod09a1061_sr_500m_202210.tif':
                    print(os.path.basename(outImage))
                    for inImage in srImages:
                        print(os.path.basename(inImage))
                    

def make_mask_example():
    qcImage = r'S:\aust\modis_surface_reflectance\MODIS_Terra_8day_MOD09A1-061_state\MOD09A1.061_sur_refl_state_500m_doy2022297_aid0001.tif'
    
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    infiles.qc = qcImage
    outfiles.mask = r'S:\aust\modis_surface_reflectance\test.tif'
    applier.apply(make_mask, infiles, outfiles, otherArgs=otherargs, controls=controls)


def make_mask(info, inputs, outputs, otherargs):
    qa = unpackqa.unpack_to_array(inputs.qc, product=MOD09A1_info)[0]
    mask = np.zeros((qa.shape[0], qa.shape[1]), dtype=np.uint8)
    mask[qa[:, :, 0] == 1] = 1 # cloud
    mask[qa[:, :, 0] == 2] = 2 # mixed
    mask[qa[:, :, 0] == 3] = 3 # assumed clear
    mask[qa[:, :, 1] == 1] = 4 # cloud shadow
    mask[qa[:, :, 4] == 1] = 5 # small cirrus
    mask[qa[:, :, 4] == 2] = 6 # average cirrus
    mask[qa[:, :, 4] == 3] = 7 # high cirrus
    mask[qa[:, :, 5] == 1] = 8 # cloud
    outputs.mask = np.array([mask])


#stackbands()
#monthly_mosaics()
#make_mask_example()