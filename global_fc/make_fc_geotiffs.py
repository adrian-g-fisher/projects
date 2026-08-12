#!/usr/bin/env python
"""

conda activate modis
                    
"""
import os, sys
import numpy as np
import glob
import xarray as xr
import rioxarray
from osgeo import gdal, ogr, osr
from rios import applier, cuiprogress
gdal.UseExceptions()


# Iterate over netcdf images and create monthly tif images
def netcdf2tif():
    inDir = r'S:\global\modis_fractional_cover\netcdf'
    outDir = r'S:\global\modis_fractional_cover\tif'
    for inFile in glob.glob(os.path.join(inDir, '*.nc')):
        outBase = os.path.basename(inFile).replace('.006.nc', '').replace('.061.nc', '').replace('.', '_')
        ds = xr.open_dataset(inFile)
        dates = ds['time'].values
        for d in dates:
            year = str(d)[0:4]
            month = str(d)[5:7]
            outFile = os.path.join(outDir, '%s%s.tif'%(outBase, month))
            if os.path.exists(outFile) is False:
                
                print(os.path.basename(outFile))
                
                ds_m = ds.sel(time="%s-%s"%(year, month)).isel(time=0)
                ds_m.rio.write_crs("ESRI:54008", inplace=True)
                ds_m.rio.to_raster(outFile)


def calcStats(info, inputs, outputs, otherargs):
    """
    This function is called from RIOS to calculate the stats image from the
    input files.
    """
    stack = np.array(inputs.fc_list).astype(np.float32)
    green_stack = stack[:, 0, :, :]
    dead_stack = stack[:, 1, :, :]
    bare_stack = stack[:, 2, :, :]
    total_stack = stack[:, 3, :, :]
    
    green_nodata = (stack[:, 0, :, :] == 255)
    green_stack = np.ma.masked_where(green_nodata == 1, green_stack)
    green_stack = green_stack.astype(float).filled(np.nan)
    dead_nodata = (stack[:, 1, :, :] == 255)
    dead_stack = np.ma.masked_where(dead_nodata == 1, dead_stack)
    dead_stack = dead_stack.astype(float).filled(np.nan)
    bare_nodata = (stack[:, 2, :, :] == 255)
    bare_stack = np.ma.masked_where(bare_nodata == 1, bare_stack)
    bare_stack = bare_stack.astype(float).filled(np.nan)
    total_nodata = (stack[:, 3, :, :] == 255)
    total_stack = np.ma.masked_where(total_nodata == 1, total_stack)
    total_stack = total_stack.astype(float).filled(np.nan)
    nodata = (np.sum(green_nodata, axis=0) == stack.shape[0])
    
    if np.isnan(green_stack).all():
        greenp05 = np.full_like(nodata, 255)
        greenp25 = np.full_like(nodata, 255)
        greenp50 = np.full_like(nodata, 255)
        greenp75 = np.full_like(nodata, 255)
        greenp95 = np.full_like(nodata, 255)
    else:
        greenP = np.nanpercentile(green_stack, [5, 25, 50, 75, 95], axis=0)
        greenp05 = greenP[0]
        greenp05[nodata == 1] = 255
        greenp25 = greenP[1]
        greenp25[nodata == 1] = 255
        greenp50 = greenP[1]
        greenp50[nodata == 1] = 255
        greenp75 = greenP[1]
        greenp75[nodata == 1] = 255
        greenp95 = greenP[1]
        greenp95[nodata == 1] = 255
    
    if np.isnan(dead_stack).all():
        deadp05 = np.full_like(nodata, 255)
        deadp25 = np.full_like(nodata, 255)
        deadp50 = np.full_like(nodata, 255)
        deadp75 = np.full_like(nodata, 255)
        deadp95 = np.full_like(nodata, 255)
    else:
        deadP = np.nanpercentile(dead_stack, [5, 25, 50, 75, 95], axis=0)
        deadp05 = deadP[0]
        deadp05[nodata == 1] = 255
        deadp25 = deadP[1]
        deadp25[nodata == 1] = 255
        deadp50 = deadP[1]
        deadp50[nodata == 1] = 255
        deadp75 = deadP[1]
        deadp75[nodata == 1] = 255
        deadp95 = deadP[1]
        deadp95[nodata == 1] = 255
    
    if np.isnan(bare_stack).all():
        barep05 = np.full_like(nodata, 255)
        barep25 = np.full_like(nodata, 255)
        barep50 = np.full_like(nodata, 255)
        barep75 = np.full_like(nodata, 255)
        barep95 = np.full_like(nodata, 255)
    else:
        bareP = np.nanpercentile(bare_stack, [5, 25, 50, 75, 95], axis=0)
        barep05 = bareP[0]
        barep05[nodata == 1] = 255
        barep25 = bareP[1]
        barep25[nodata == 1] = 255
        barep50 = bareP[1]
        barep50[nodata == 1] = 255
        barep75 = bareP[1]
        barep75[nodata == 1] = 255
        barep95 = bareP[1]
        barep95[nodata == 1] = 255

    if np.isnan(total_stack).all():
        totalp05 = np.full_like(nodata, 255)
        totalp25 = np.full_like(nodata, 255)
        totalp50 = np.full_like(nodata, 255)
        totalp75 = np.full_like(nodata, 255)
        totalp95 = np.full_like(nodata, 255)
    else:
        totalP = np.nanpercentile(total_stack, [5, 25, 50, 75, 95], axis=0)
        totalp05 = totalP[0]
        totalp05[nodata == 1] = 255
        totalp25 = totalP[1]
        totalp25[nodata == 1] = 255
        totalp50 = totalP[1]
        totalp50[nodata == 1] = 255
        totalp75 = totalP[1]
        totalp75[nodata == 1] = 255
        totalp95 = totalP[1]
        totalp95[nodata == 1] = 255
    
    outputs.p05 = np.array([greenp05, deadp05, barep05, totalp05]).astype(np.uint8)
    outputs.p25 = np.array([greenp25, deadp25, barep25, totalp25]).astype(np.uint8)
    outputs.p50 = np.array([greenp50, deadp50, barep50, totalp50]).astype(np.uint8)
    outputs.p75 = np.array([greenp75, deadp75, barep75, totalp75]).astype(np.uint8)
    outputs.p95 = np.array([greenp95, deadp95, barep95, totalp95]).astype(np.uint8)


def calculate_percentiles():
    
    with open('S:/global/modis_fractional_cover/modis_hv_countries.txt', 'r') as f:
        hvCountries = f.read().splitlines()[1:]
    
    inDir = r'S:\global\modis_fractional_cover\tif'
    outDir = r'S:\global\modis_fractional_cover\percentiles'
    imageList = glob.glob(os.path.join(inDir, "*.tif"))
    hvList = np.array([os.path.basename(i).split("_")[-2] for i in imageList])
    imageList = np.array(imageList)
    hv_unique = np.unique(hvList)
    for hv in hvCountries:
        if hv in hv_unique:
            p05 = os.path.join(outDir, r'p05/FC_Monthly_Medoid_v310_MCD43A4_%s_p05.tif'%hv)
            p25 = os.path.join(outDir, r'p25/FC_Monthly_Medoid_v310_MCD43A4_%s_p25.tif'%hv)
            p50 = os.path.join(outDir, r'p50/FC_Monthly_Medoid_v310_MCD43A4_%s_p50.tif'%hv)
            p75 = os.path.join(outDir, r'p75/FC_Monthly_Medoid_v310_MCD43A4_%s_p75.tif'%hv)
            p95 = os.path.join(outDir, r'p95/FC_Monthly_Medoid_v310_MCD43A4_%s_p95.tif'%hv)
            
            if all(os.path.isfile(f) for f in [p05, p25, p50, p75, p95]) is True:
                print("Completed %s"%hv)
            
            else:
                print("Processing %s"%hv)
                hv_images = list(imageList[hvList == hv])
                infiles = applier.FilenameAssociations()
                infiles.fc_list = hv_images
                outfiles = applier.FilenameAssociations()
                outfiles.p05 = p05
                outfiles.p25 = p25
                outfiles.p50 = p50
                outfiles.p75 = p75
                outfiles.p95 = p95
                otherargs = applier.OtherInputs()
                controls = applier.ApplierControls()
                controls.setWindowXsize(256)
                controls.setWindowYsize(256)
                controls.setStatsIgnore(255)
                controls.setCalcStats(True)
                controls.setOutputDriverName("GTiff")
                controls.setReferenceImage(hv_images[0])
                controls.setResampleMethod('near')
                controls.setLayerNames(['Photosynthetic vegetation', 'Non-photosynthetic vegetation', 'Bare soil', 'Total cover'])
                controls.setProgress(cuiprogress.CUIProgressBar()) 
                applier.apply(calcStats, infiles, outfiles, otherArgs=otherargs, controls=controls)


def merge_tiles_globally():
    inDir = r'S:\global\modis_fractional_cover\percentiles'
    for p in ['p05', 'p25', 'p50', 'p75', 'p95']:
        imageList = glob.glob(os.path.join(inDir, "p/*.tif"))
        outFile = os.path.join(inDir, 'FC_Monthly_Medoid_v310_MCD43A4_global_%s.tif'%p)
        if os.path.exists(outFile) is False:
            outVrt = outFile.replace('.tif', '.vrt')
            outds = gdal.BuildVRT(outVrt, imageList.tolist())
            outds = gdal.Translate(outFile, outds)
            bandnames = ['Photosynthetic vegetation', 'Non-photosynthetic vegetation', 'Bare soil', 'Total cover']
            for i in range(4):
                band = outds.GetRasterBand(i+1)
                band.SetDescription(bandnames[i])
            gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
            outds.BuildOverviews()
            outds = None
            os.remove(outVrt)


#netcdf2tif()
calculate_percentiles()
#merge_tiles_globally()