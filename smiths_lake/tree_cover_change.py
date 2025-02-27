#!/usr/bin/env python
"""

Mapping regrowth of trees on Horse point 1989-2022

Total area = 540000 m^2
1989 = 179900 m^2 
1997 = 206229 m^2 
2002 = 220085 m^2
2009 = 241772 m^2
2013 = 281217 m^2 (ALS)
2017 = 330016 m^2 
2022 = 377402 m^2
"""

import os
import sys
import argparse
import glob
import numpy as np
from rios import applier
from scipy import ndimage
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt


def binary_trees(info, inputs, outputs, otherargs):
    binary = np.where(inputs.chm[0] < 2, 0, 1)
    binary[inputs.aoi[0] == 0] = 0
    outputs.binary = np.array([binary]).astype(np.uint8)
    

def make_treemap(chm_image, aoi_shapefile):
    infiles = applier.FilenameAssociations()
    infiles.chm = chm_image
    infiles.aoi = aoi_shapefile
    outfiles = applier.FilenameAssociations()
    outfiles.binary = chm_image.replace('_chm.tif', '_treemap.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    applier.apply(binary_trees, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.binary)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = chm_image.replace('_chm.tif', '_treemap')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.binary)
    

def make_trees_2013():
    aoi_shapefile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\smiths_lake\horsepoint_aoi.shp'
    chm_image = r'S:\smiths_lake\lidar\smiths_lake_201306_chm.tif'
    make_treemap(chm_image, aoi_shapefile)
    

def update_trees(info, inputs, outputs, otherargs):
    b = inputs.bw[0]
    b[inputs.trees_2013[0] == 0] = 255 
    b[b <= 130] = 1
    b[b > 130] = 0
    
    # Remove single pixels
    b = ndimage.binary_erosion(b)
    b = ndimage.binary_dilation(b)
    
    # Fill holes
    b = ndimage.binary_fill_holes(b)
    
    outputs.trees_1989 = np.array([b]).astype(np.uint8)


def make_trees_1989():
    bw_image = r'S:\smiths_lake\imagery\airphotos\mosaics\af10bw_n3679r007f0032_19890831_ba1m6_horsepoint.tif'
    trees_2013 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_201306_treemap.shp'
    infiles = applier.FilenameAssociations()
    infiles.bw = bw_image
    infiles.trees_2013 = trees_2013
    outfiles = applier.FilenameAssociations()
    outfiles.trees_1989 = trees_2013.replace('_201306_treemap.shp', '_198908_treemap.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    applier.apply(update_trees, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_1989)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_2013.replace('_201306_treemap.shp', '_198908_treemap')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_1989)


def refine_1989(info, inputs, outputs, otherargs):
    b = inputs.trees_1989[0]
    b[inputs.trees_max[0] != 2] = 0 
    outputs.trees_refined = np.array([b]).astype(np.uint8)
    
    
def refine_trees_1989():
    bw_image = r'S:\smiths_lake\imagery\airphotos\mosaics\af10bw_n3679r007f0032_19890831_ba1m6_horsepoint.tif'
    trees_1989 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_198908_treemap.shp'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    infiles = applier.FilenameAssociations()
    infiles.bw = bw_image
    infiles.trees_1989 = trees_1989
    infiles.trees_max = trees_max
    outfiles = applier.FilenameAssociations()
    outfiles.trees_refined = trees_1989.replace('.shp', '_refined.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(refine_1989, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_refined)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_1989.replace('.shp', '_refined')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_refined)
    

def map_trees_2022(info, inputs, outputs, otherargs):
    green = inputs.col[1]
    green[inputs.trees_max[0] != 2] = 255 
    green[green <= 70] = 1
    green[green > 70] = 0
    
    # Remove single pixels
    green = ndimage.binary_erosion(green)
    green = ndimage.binary_dilation(green)
    
    # Add back other polygons
    green[inputs.trees_max[0] == 1] = 1
    
    outputs.trees_2022 = np.array([green]).astype(np.uint8)


def make_trees_2022():
    col_image = r'S:\smiths_lake\imagery\google_earth\airbus_20220503_m6.tif'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_max = trees_max
    outfiles = applier.FilenameAssociations()
    outfiles.trees_2022 = trees_max.replace('_maxextent_noholes.shp', '_202205_treemap.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(map_trees_2022, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_2022)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_max.replace('_maxextent_noholes.shp', '_202205_treemap')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_2022)
    
    
def map_trees_2017(info, inputs, outputs, otherargs):
    green = inputs.col[1]
    green[inputs.trees_max[0] != 2] = 255 
    green[green <= 70] = 1
    green[green > 70] = 0
    
    # Remove single pixels
    green = ndimage.binary_erosion(green)
    green = ndimage.binary_dilation(green)
    
    # Add back other polygons
    green[inputs.trees_max[0] == 1] = 1
    
    # Add back northern coast
    green[inputs.trees_coast[0] == 1] = 1
    
    outputs.trees_2017 = np.array([green]).astype(np.uint8)


def make_trees_2017():
    col_image = r'S:\smiths_lake\imagery\google_earth\maxar_20170428_m6.tif'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    trees_coast = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_northerncoast.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_max = trees_max
    infiles.trees_coast = trees_coast
    outfiles = applier.FilenameAssociations()
    outfiles.trees_2017 = trees_max.replace('_maxextent_noholes.shp', '_201704_treemap.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(map_trees_2017, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_2017)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_max.replace('_maxextent_noholes.shp', '_201704_treemap')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_2017)


def map_trees_1997(info, inputs, outputs, otherargs):
    green = inputs.col[1]
    green[inputs.trees_max[0] != 2] = 255 
    green[green <= 100] = 1
    green[green > 100] = 0
    
    # Remove single pixels
    green = ndimage.binary_erosion(green)
    green = ndimage.binary_dilation(green)
    
    # Add back other polygons
    green[inputs.trees_max[0] == 1] = 1
    green[inputs.trees_1989[0] == 1] = 1
    
    outputs.trees_1997 = np.array([green]).astype(np.uint8)


def make_trees_1997():
    col_image = r'S:\smiths_lake\imagery\airphotos\rectified\horsepoint_airphoto_1997_4374_11_076_m6.tif'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    trees_1989 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_198908_treemap.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_max = trees_max
    infiles.trees_1989 = trees_1989
    outfiles = applier.FilenameAssociations()
    outfiles.trees_1997 = trees_max.replace('_maxextent_noholes.shp', '_199708_treemap.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(map_trees_1997, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_1997)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_max.replace('_maxextent_noholes.shp', '_199708_treemap')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_1997)


def map_trees_2002(info, inputs, outputs, otherargs):
    green = inputs.col[1]
    green[inputs.trees_max[0] != 2] = 255 
    green[green <= 70] = 1
    green[green > 70] = 0
    
    # Remove shadows
    green[inputs.shadows[0] == 1] = 0
    
    # Remove single pixels
    green = ndimage.binary_erosion(green)
    green = ndimage.binary_dilation(green)
    
    # Add back other polygons
    green[inputs.trees_max[0] == 1] = 1
    green[inputs.trees_1989[0] == 1] = 1
    
    outputs.trees_2002 = np.array([green]).astype(np.uint8)


def make_trees_2002():
    col_image = r'S:\smiths_lake\imagery\google_earth\maxar_20021123_m6.tif'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    trees_1989 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_198908_treemap.shp'
    shadows = r'S:\smiths_lake\imagery\google_earth\maxar_20021123_m6_cloudshadows.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_max = trees_max
    infiles.trees_1989 = trees_1989
    infiles.shadows = shadows
    outfiles = applier.FilenameAssociations()
    outfiles.trees_2002 = trees_1989.replace('_198908_treemap.shp', '_200211_treemap.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(map_trees_2002, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_2002)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_1989.replace('_198908_treemap.shp', '_200211_treemap')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_2002)


def map_trees_2009(info, inputs, outputs, otherargs):
    green = inputs.col[1]
    green[inputs.trees_max[0] != 2] = 255 
    green[green <= 105] = 1
    green[green > 105] = 0
    
    # Remove single pixels
    green = ndimage.binary_erosion(green)
    green = ndimage.binary_dilation(green)
    
    # Add back other polygons
    green[inputs.trees_max[0] == 1] = 1
    green[inputs.trees_2002[0] == 1] = 1
    
    outputs.trees_2009 = np.array([green]).astype(np.uint8)


def make_trees_2009():
    col_image = r'S:\smiths_lake\imagery\google_earth\maxar_20091204_m6.tif'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    trees_2002 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_200211_treemap.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_max = trees_max
    infiles.trees_2002 = trees_2002
    outfiles = applier.FilenameAssociations()
    outfiles.trees_2009 = trees_2002.replace('_200211_treemap.shp', '_200912_treemap.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(map_trees_2009, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_2009)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_2002.replace('_200211_treemap.shp', '_200912_treemap')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_2009)


def refine_1997(info, inputs, outputs, otherargs):
    b = inputs.trees_1997[0]
    b[inputs.trees_max[0] != 2] = 0 
    outputs.trees_refined = np.array([b]).astype(np.uint8)
    
    
def refine_trees_1997():
    col_image = r'S:\smiths_lake\imagery\airphotos\rectified\horsepoint_airphoto_1997_4374_11_076_m6.tif'
    trees_1997 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_199708_treemap.shp'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_1997 = trees_1997
    infiles.trees_max = trees_max
    outfiles = applier.FilenameAssociations()
    outfiles.trees_refined = trees_1997.replace('.shp', '_refined.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(refine_1997, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_refined)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_1997.replace('.shp', '_refined')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_refined)


def refine_2002(info, inputs, outputs, otherargs):
    b = inputs.trees_2002[0]
    b[inputs.trees_max[0] != 2] = 0 
    outputs.trees_refined = np.array([b]).astype(np.uint8)
    
    
def refine_trees_2002():
    col_image = r'S:\smiths_lake\imagery\google_earth\maxar_20021123_m6.tif'
    trees_2002 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_200211_treemap.shp'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_2002 = trees_2002
    infiles.trees_max = trees_max
    outfiles = applier.FilenameAssociations()
    outfiles.trees_refined = trees_2002.replace('.shp', '_refined.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(refine_2002, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_refined)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_2002.replace('.shp', '_refined')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_refined)


def refine_2009(info, inputs, outputs, otherargs):
    b = inputs.trees_2009[0]
    b[inputs.trees_max[0] != 2] = 0 
    outputs.trees_refined = np.array([b]).astype(np.uint8)
    
    
def refine_trees_2009():
    col_image = r'S:\smiths_lake\imagery\google_earth\maxar_20091204_m6.tif'
    trees_2009 = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_200912_treemap.shp'
    trees_max = r'S:\smiths_lake\horsepoint_tree_maps\horsepoint_maxextent_noholes.shp'
    infiles = applier.FilenameAssociations()
    infiles.col = col_image
    infiles.trees_2009 = trees_2009
    infiles.trees_max = trees_max
    outfiles = applier.FilenameAssociations()
    outfiles.trees_refined = trees_2009.replace('.shp', '_refined.tif')
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setThematic(True)
    controls.setWindowSize(2048, 2048)
    controls.setBurnAttribute("DN")
    applier.apply(refine_2009, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
    src_ds = gdal.Open(outfiles.trees_refined)
    srcband = src_ds.GetRasterBand(1)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    dst_layername = trees_2009.replace('.shp', '_refined')
    if os.path.exists(dst_layername + ".shp"):
        os.remove(dst_layername + ".shp")
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername + ".shp")
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)
    newField = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(newField)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")
    gdal.Polygonize(srcband, srcband, dst_layer, dst_field, [], callback=None)
    dst_ds.Destroy()
    src_ds.Destroy()
    os.remove(outfiles.trees_refined)
    
    
# Hardcode steps
#make_trees_2013()
#make_trees_1989()
#make_trees_2022()
#make_trees_2017()
#make_trees_1997()
#make_trees_2002()
#make_trees_2009()
#refine_trees_1989()
#refine_trees_1997()
#refine_trees_2002()
refine_trees_2009()