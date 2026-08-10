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
    
    p = 5
    pGreen = np.nanpercentile(green_stack, p, axis=0)
    pGreen[nodata == 1] = 255
    pDead = np.nanpercentile(dead_stack, p, axis=0)
    pDead[nodata == 1] = 255
    pBare = np.nanpercentile(bare_stack, p, axis=0)
    pBare[nodata == 1] = 255
    pTotal = np.nanpercentile(total_stack, p, axis=0)
    pTotal[nodata == 1] = 255
    outputs.p05 = np.array([pGreen, pDead, pBare, pTotal]).astype(np.uint8)
    
    p = 25
    pGreen = np.nanpercentile(green_stack, p, axis=0)
    pGreen[nodata == 1] = 255
    pDead = np.nanpercentile(dead_stack, p, axis=0)
    pDead[nodata == 1] = 255
    pBare = np.nanpercentile(bare_stack, p, axis=0)
    pBare[nodata == 1] = 255
    pTotal = np.nanpercentile(total_stack, p, axis=0)
    pTotal[nodata == 1] = 255
    outputs.p25 = np.array([pGreen, pDead, pBare, pTotal]).astype(np.uint8)
    
    p = 50
    pGreen = np.nanpercentile(green_stack, p, axis=0)
    pGreen[nodata == 1] = 255
    pDead = np.nanpercentile(dead_stack, p, axis=0)
    pDead[nodata == 1] = 255
    pBare = np.nanpercentile(bare_stack, p, axis=0)
    pBare[nodata == 1] = 255
    pTotal = np.nanpercentile(total_stack, p, axis=0)
    pTotal[nodata == 1] = 255
    outputs.p50 = np.array([pGreen, pDead, pBare, pTotal]).astype(np.uint8)
    
    p = 75
    pGreen = np.nanpercentile(green_stack, p, axis=0)
    pGreen[nodata == 1] = 255
    pDead = np.nanpercentile(dead_stack, p, axis=0)
    pDead[nodata == 1] = 255
    pBare = np.nanpercentile(bare_stack, p, axis=0)
    pBare[nodata == 1] = 255
    pTotal = np.nanpercentile(total_stack, p, axis=0)
    pTotal[nodata == 1] = 255
    outputs.p75 = np.array([pGreen, pDead, pBare, pTotal]).astype(np.uint8)

    p = 95
    pGreen = np.nanpercentile(green_stack, p, axis=0)
    pGreen[nodata == 1] = 255
    pDead = np.nanpercentile(dead_stack, p, axis=0)
    pDead[nodata == 1] = 255
    pBare = np.nanpercentile(bare_stack, p, axis=0)
    pBare[nodata == 1] = 255
    pTotal = np.nanpercentile(total_stack, p, axis=0)
    pTotal[nodata == 1] = 255
    outputs.p95 = np.array([pGreen, pDead, pBare, pTotal]).astype(np.uint8)


def calculate_percentiles():
    inDir = r'S:\global\modis_fractional_cover\tif'
    outDir = r'S:\global\modis_fractional_cover\percentiles'
    imageList = glob.glob(os.path.join(inDir, "*.tif"))
    hvList = np.array([os.path.basename(i).split("_")[-2] for i in imageList])
    imageList = np.array(imageList)
    hv_unique = np.unique(hvList)
    for hv in hv_unique:
        hv_images = list(imageList[hvList == hv])
        
        # geotransforms can be different due to rounding of pixel sizes!
        # ref_gt = None
        # for hvImage in hv_images:
            # ds = gdal.Open(hvImage)
            # gt = ds.GetGeoTransform()
            # ds = None
            # if ref_gt == None:
                # ref_gt = gt
            # else:
                # if gt != ref_gt:
                    # print(os.path.basename(hvImage))
                    # print(ref_gt)
                    # print(gt)
        
        infiles = applier.FilenameAssociations()
        infiles.fc_list = hv_images
        outfiles = applier.FilenameAssociations()
        outfiles.p05 = os.path.join(outDir, r'p05/FC_Monthly_Medoid_v310_MCD43A4_%s_p05.tif'%hv)
        outfiles.p25 = os.path.join(outDir, r'p25/FC_Monthly_Medoid_v310_MCD43A4_%s_p25.tif'%hv)
        outfiles.p50 = os.path.join(outDir, r'p50/FC_Monthly_Medoid_v310_MCD43A4_%s_p50.tif'%hv)
        outfiles.p75 = os.path.join(outDir, r'p75/FC_Monthly_Medoid_v310_MCD43A4_%s_p75.tif'%hv)
        outfiles.p95 = os.path.join(outDir, r'p95/FC_Monthly_Medoid_v310_MCD43A4_%s_p95.tif'%hv)
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
        print('Created percentiles for %s'%hv)


# Merge dates together
# I need to change this to merge the percentiles
def merge_dates_globally():
    inDir = r'S:\global\modis_fractional_cover\tif'
    outDir = r'S:\global\modis_fractional_cover\global'
    imageList = glob.glob(os.path.join(inDir, "*.tif"))
    dateList = np.array([os.path.basename(i).split("_")[-1].replace(".tif", "") for i in imageList])
    imageList = np.array(imageList)
    date_unique = np.unique(dateList)
    for date in date_unique:
        print(date)
        dateImages = imageList[dateList == date]
        outFile = os.path.join(outDir, 'FC_Monthly_Medoid_v310_MCD43A4_global_%s.tif'%date)
        if os.path.exists(outFile) is False:
            outVrt = outFile.replace('.tif', '.vrt')
            outds = gdal.BuildVRT(outVrt, dateImages.tolist())
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
#merge_dates_globally()