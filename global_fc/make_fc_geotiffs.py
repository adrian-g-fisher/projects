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
            
            if all(os.path.isfile(f) for f in [p05, p25, p50, p75, p95]) is False:
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
    inDir = r'S:/global/modis_fractional_cover/percentiles'
    for p in ['p05', 'p25', 'p50', 'p75', 'p95']:
        imageList = glob.glob(os.path.join(inDir, "%s/*.tif"%p))
        outFile = os.path.join(inDir, 'FC_Monthly_Medoid_v310_MCD43A4_global_%s.tif'%p)
        if os.path.exists(outFile) is False:
            outVrt = outFile.replace('.tif', '.vrt')
            outds = gdal.BuildVRT(outVrt, list(imageList))
            outds = gdal.Translate(outFile, outds)
            bandnames = ['Photosynthetic vegetation', 'Non-photosynthetic vegetation', 'Bare soil', 'Total cover']
            for i in range(4):
                band = outds.GetRasterBand(i+1)
                band.SetDescription(bandnames[i])
            gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
            outds.BuildOverviews()
            outds = None
            os.remove(outVrt)


def fixNodata(info, inputs, outputs, otherargs):
    """
    This fixes the nodata values.
    """
    PV = inputs.FC[0]
    NPV = inputs.FC[1]
    BS = inputs.FC[2]
    TV = inputs.FC[3]
    extraNodata = (PV == 1) & (NPV == 1) & (BS == 1)
    PV[extraNodata] = 255
    NPV[extraNodata] = 255
    BS[extraNodata] = 255
    TV[extraNodata] = 255
    outputs.FC = np.array([PV, NPV, BS, TV]).astype(np.uint8)
    

def fix_nodata():
    inDir = r'S:/global/modis_fractional_cover/percentiles'
    for p in ['p05', 'p25', 'p50', 'p75', 'p95']:
        inFile = os.path.join(inDir, 'FC_Monthly_Medoid_v310_MCD43A4_global_%s.tif'%p)
        outFile = os.path.join(inDir, 'FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_%s.tif'%p)
        
        print(os.path.basename(outFile))
        
        infiles = applier.FilenameAssociations()
        infiles.FC = inFile
        outfiles = applier.FilenameAssociations()
        outfiles.FC = outFile
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        controls.setWindowXsize(256)
        controls.setWindowYsize(256)
        controls.setStatsIgnore(255)
        controls.setCalcStats(True)
        controls.setOutputDriverName("GTiff")
        controls.setLayerNames(['Photosynthetic vegetation', 'Non-photosynthetic vegetation', 'Bare soil', 'Total cover'])
        controls.setProgress(cuiprogress.CUIProgressBar()) 
        applier.apply(fixNodata, infiles, outfiles, otherArgs=otherargs, controls=controls)


def fixNodata(info, inputs, outputs, otherargs):
    """
    Fixes the nodata problem in the aridity data
    """
    aridity = inputs.aridity[0]
    africa = inputs.africa[0]
    aridity[(africa == 1) & (aridity == 0)] = 1
    outputs.aridity = np.array([aridity])


def resampleImage(info, inputs, outputs, otherargs):
    """
    Resamples aridity to modis
    """
    aridity = inputs.aridity[0].astype(np.float32) * 0.0001
    nodata = (inputs.fc[0] == 255) & (inputs.fc[1] == 255) & (inputs.fc[2] == 255)
    aridity[nodata == 1] = -999
    outputs.aridity = np.array([aridity]).astype(np.float32)


def resample_aridity():
    
    # First fix the nodata values in Africa
    infiles = applier.FilenameAssociations()
    infiles.aridity = 'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr.tif'
    infiles.africa = 'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/africa_nodata.shp'
    outfiles = applier.FilenameAssociations()
    outfiles.aridity = 'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr_nodata_fixed.tif'
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setWindowXsize(256)
    controls.setWindowYsize(256)
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar()) 
    applier.apply(fixNodata, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    # Now resample using gdal.warp
    inImage = 'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr_nodata_fixed.tif'
    refImage = 'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p50.tif'
    outImage = 'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr_gdalwarp.tif'
    ref_ds = gdal.Open(refImage, gdal.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    ref_width = ref_ds.RasterXSize
    ref_height = ref_ds.RasterYSize
    ref_ds = None
    min_x = ref_geotrans[0]
    max_y = ref_geotrans[3]
    max_x = min_x + ref_geotrans[1] * ref_width
    min_y = max_y + ref_geotrans[5] * ref_height
    ref_output_bounds = [min_x, min_y, max_x, max_y]
    warp_options = gdal.WarpOptions(format='GTiff',
                                    dstSRS=ref_proj,
                                    outputBounds=ref_output_bounds,
                                    xRes=ref_geotrans[1],
                                    yRes=abs(ref_geotrans[5]),
                                    resampleAlg='bilinear',
                                    creationOptions=['COMPRESS=DEFLATE'])
    gdal.Warp(outImage, inImage, options=warp_options)
    
    # Now set nodata to mask oceans and lakes
    infiles = applier.FilenameAssociations()
    infiles.aridity = 'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr_gdalwarp.tif'
    infiles.fc = 'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p50.tif'
    outfiles = applier.FilenameAssociations()
    outfiles.aridity = 'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr_modis_masked.tif'
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setWindowXsize(256)
    controls.setWindowYsize(256)
    controls.setStatsIgnore(-999)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar()) 
    applier.apply(resampleImage, infiles, outfiles, otherArgs=otherargs, controls=controls)


def resamplePop(info, inputs, outputs, otherargs):
    """
    Removes water from resampled pop
    """
    pop = inputs.pop[0].astype(np.float32)
    pop[pop < 0] = 0
    nodata = (inputs.fc[0] == 255) & (inputs.fc[1] == 255) & (inputs.fc[2] == 255)
    pop[nodata == 1] = -999
    outputs.pop = np.array([pop]).astype(np.float32)


def resample_population():
    
    # Resample using gdal.warp
    inImage = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec.tif'
    refImage = 'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p50.tif'
    outImage = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec_sinusoidal.tif'
    ref_ds = gdal.Open(refImage, gdal.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    ref_width = ref_ds.RasterXSize
    ref_height = ref_ds.RasterYSize
    ref_ds = None
    min_x = ref_geotrans[0]
    max_y = ref_geotrans[3]
    max_x = min_x + ref_geotrans[1] * ref_width
    min_y = max_y + ref_geotrans[5] * ref_height
    ref_output_bounds = [min_x, min_y, max_x, max_y]
    warp_options = gdal.WarpOptions(format='GTiff',
                                    dstSRS=ref_proj,
                                    outputBounds=ref_output_bounds,
                                    xRes=ref_geotrans[1],
                                    yRes=abs(ref_geotrans[5]),
                                    resampleAlg='bilinear',
                                    creationOptions=['COMPRESS=DEFLATE'])
    gdal.Warp(outImage, inImage, options=warp_options)

    # Now set nodata to mask oceans and lakes
    infiles = applier.FilenameAssociations()
    infiles.pop = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec_sinusoidal.tif'
    infiles.fc = 'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p50.tif'
    outfiles = applier.FilenameAssociations()
    outfiles.pop = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec_sinusoidal_masked.tif'
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setWindowXsize(256)
    controls.setWindowYsize(256)
    controls.setStatsIgnore(-999)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar()) 
    applier.apply(resamplePop, infiles, outfiles, otherArgs=otherargs, controls=controls)

    
def make_binary_salt(info, inputs, outputs, otherargs):
    """
    Makes a binary mask for salt lakes
    2 = Saline lake
    32 = Salt pan, saline/brackish wetland
    """
    lakes = inputs.lakes[0]
    salt = np.where((lakes == 2) | (lakes == 32), 1, 0)
    outputs.salt = np.array([salt]).astype(np.uint8)

    
def make_saltlake_mask():

    # Create binary saltlake mask image
    infiles = applier.FilenameAssociations()
    infiles.lakes = 'S:/global/GLWD_v2_0/GLWD_v2_0_combined_classes/GLWD_v2_0_main_class.tif'
    outfiles = applier.FilenameAssociations()
    outfiles.salt = 'S:/global/GLWD_v2_0/GLWD_v2_0_combined_classes/GLWD_v2_0_saltlakes.tif'
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setWindowXsize(256)
    controls.setWindowYsize(256)
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setProgress(cuiprogress.CUIProgressBar()) 
    applier.apply(make_binary_salt, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    # Resample using arcgis - gdal.warp is doing something weird


def fix_saltlakes():
    inImage = 'S:/global/GLWD_v2_0/GLWD_v2_0_combined_classes/GLWD_v2_0_saltlakes_sinusoidal_clip.tif'
    refImage = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec_sinusoidal_masked.tif'
    outImage = 'S:/global/GLWD_v2_0/GLWD_v2_0_combined_classes/GLWD_v2_0_saltlakes_sinusoidal_clip_fixed.tif'
    ref_ds = gdal.Open(refImage, gdal.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    ref_width = ref_ds.RasterXSize
    ref_height = ref_ds.RasterYSize
    ref_ds = None
    min_x = ref_geotrans[0]
    max_y = ref_geotrans[3]
    max_x = min_x + ref_geotrans[1] * ref_width
    min_y = max_y + ref_geotrans[5] * ref_height
    ref_output_bounds = [min_x, min_y, max_x, max_y]
    warp_options = gdal.WarpOptions(format='GTiff',
                                    dstSRS=ref_proj,
                                    outputBounds=ref_output_bounds,
                                    xRes=ref_geotrans[1],
                                    yRes=abs(ref_geotrans[5]),
                                    resampleAlg='bilinear',
                                    creationOptions=['COMPRESS=DEFLATE'])
    gdal.Warp(outImage, inImage, options=warp_options)


def rasterise_drylands():
    input_vector = r'S:/global/Drylands_dataset_2007/drylands_UNCCD_CBD_july2014_sinusoidal.shp'
    output_raster = r'S:/global/Drylands_dataset_2007/drylands_UNCCD_CBD_july2014_sinusoidal.tif'
    refImage = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec_sinusoidal_masked.tif'
    ref_ds = gdal.Open(refImage, gdal.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    ref_width = ref_ds.RasterXSize
    ref_height = ref_ds.RasterYSize
    ref_ds = None
    min_x = ref_geotrans[0]
    max_y = ref_geotrans[3]
    max_x = min_x + ref_geotrans[1] * ref_width
    min_y = max_y + ref_geotrans[5] * ref_height
    ref_output_bounds = [min_x, min_y, max_x, max_y]
    driver = gdal.GetDriverByName("GTiff")
    target_ds = driver.Create(output_raster, ref_width, ref_height, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((min_x, ref_geotrans[1], 0.0, max_y, 0.0, ref_geotrans[5]))
    target_ds.SetProjection(ref_proj)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)
    vec_ds = ogr.Open(input_vector)
    layer = vec_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, options=['ATTRIBUTE=HIX_ZONE'])
    target_ds = None
    vec_ds = None


def rasterise_continents():
    input_vector = r'S:/global/continents/World_Continents_sinusoidal.shp'
    output_raster = r'S:/global/continents/World_Continents_sinusoidal.tif'
    refImage = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec_sinusoidal_masked.tif'
    ref_ds = gdal.Open(refImage, gdal.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    ref_width = ref_ds.RasterXSize
    ref_height = ref_ds.RasterYSize
    ref_ds = None
    min_x = ref_geotrans[0]
    max_y = ref_geotrans[3]
    max_x = min_x + ref_geotrans[1] * ref_width
    min_y = max_y + ref_geotrans[5] * ref_height
    ref_output_bounds = [min_x, min_y, max_x, max_y]
    driver = gdal.GetDriverByName("GTiff")
    target_ds = driver.Create(output_raster, ref_width, ref_height, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((min_x, ref_geotrans[1], 0.0, max_y, 0.0, ref_geotrans[5]))
    target_ds.SetProjection(ref_proj)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)
    vec_ds = ogr.Open(input_vector)
    layer = vec_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, options=['ATTRIBUTE=OBJECTID_1'])
    target_ds = None
    vec_ds = None


def fix_proj():
    infiles = [r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p05.tif',
               r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p25.tif',
               r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p50.tif',
               r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p75.tif',
               r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p95.tif']
    refImage = r'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr_modis_masked.tif'
    ref_ds = gdal.Open(refImage, gdal.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_ds = None
    for infile in infiles:
        ds = gdal.Open(infile, gdal.GA_Update)
        ds.SetProjection(ref_proj)
        ds = None


#netcdf2tif()
#calculate_percentiles()
#merge_tiles_globally()
#fix_nodata()
#resample_aridity()
#resample_population()
#make_saltlake_mask()
#fix_saltlakes()
#rasterise_drylands()
#rasterise_continents()
#fix_proj()