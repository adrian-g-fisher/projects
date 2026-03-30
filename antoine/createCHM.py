#!/usr/bin/env python
"""
This makes a digital surface model (DSM), digital elevation model (DEM), and
canopy height model (CHM) from LAZ file point clouds that have ground points
already classified. It processes all the data collected for the grazing study.

It uses the pixelgrid from the mosaic image file so pixels line up.

This needs:
conda create -n lidar rios scipy numba rios::pynninterp laspy lazrs-python

"""

import os
import sys
import glob
import numpy
import argparse
import laspy
import shutil
import pynninterp
from numba import jit
from osgeo import gdal
from osgeo import osr
from scipy import ndimage
from scipy import interpolate

gdal.UseExceptions()


def process_laz(lazFile, mosaic):
    """
    Main function to create a digital surface model, digital elevation model,
    and canopy height model. It resamples the mosiac image into 5 mm pixels 
    and creates the model file using the same pixelgrid.
    """
    
    # First reproject the mosaic image into a new image with square 5 mm pixels
    mosaic_5mm = lazFile.replace('_classified.laz', '_mosaic_5mm.tif')
    if os.path.exists(mosaic_5mm) is False:
        kwargs = {'xRes': 0.005, 'yRes': 0.005, 'resampleAlg': 'bilinear',
                  'targetAlignedPixels': True, 'srcNodata': 65535}
        ds = gdal.Warp(mosaic_5mm, mosaic, **kwargs)
        ds = None
    
    # Create names for output files
    dsmFile = mosaic_5mm.replace('_mosaic_5mm.tif', '_dsm_5mm.tif')
    demFile = mosaic_5mm.replace('_mosaic_5mm.tif', '_dem_5mm.tif')
    chmFile = mosaic_5mm.replace('_mosaic_5mm.tif', '_chm_5mm.tif')
    
    # Read in mosaic image and get image information
    ds = gdal.Open(mosaic_5mm)
    gt = ds.GetGeoTransform()
    pixelSize = gt[1]
    rows  = ds.RasterYSize
    columns  = ds.RasterXSize
    minX = gt[0]
    maxX = gt[0] + columns * pixelSize
    minY = gt[3] + rows * gt[5]
    maxY = gt[3]
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = int(proj.GetAttrValue('AUTHORITY', 1))
    tileXsize = maxX - minX
    tileYsize = maxY - minY
    nRows = int(numpy.ceil(tileYsize / pixelSize))
    nCols = int(numpy.ceil(tileXsize / pixelSize))
    pxlCoords = get_uneven_grid(nCols, nRows, minX, maxY, pixelSize)

    # Read in the point cloud
    with laspy.open(lazFile) as fh:
        las = fh.read()
    points = numpy.rec.fromarrays([las.x, las.y, las.z, las.classification],
                                   names=['X', 'Y', 'Z', 'CLASSIFICATION'],
                                   formats = ['<f8', '<f8', '<f8', 'u1'])
    
    # Remove points outside the mosaic grid
    goodPoints = (points['X'] > minX) & (points['X'] < maxX) & (points['Y'] > minY) & (points['Y'] < maxY)
    points = points[goodPoints]
    
    # Create digital surface model
    (x, y, z) = (points['X'], points['Y'], points['Z'])   
    (row, col) = xyToRowCol(x, y, minX, maxY, pixelSize)    
    nullVal = -999
    maxHeight = numpy.zeros((nRows, nCols), dtype=numpy.float32) + nullVal
    maxHeight = maxGridding(maxHeight, row, col, z)
    nullArray = (maxHeight == nullVal)
    xVals = pxlCoords[0][~nullArray]
    yVals = pxlCoords[1][~nullArray]
    zVals = maxHeight[~nullArray]
    dsm = interpGrid(xVals, yVals, zVals, pxlCoords)
    dsm[numpy.isnan(dsm)] = nullVal
    dsm[~nullArray] = maxHeight[~nullArray]
    dsm = numpy.round(dsm.astype(numpy.float32), 3)
    writeImage(dsm, dsmFile, driver='GTiff', tlx=minX, tly=maxY,
               binsize=pixelSize, epsg=epsg, nullVal=nullVal)

    # Create digital elevation model
    ground = (points["CLASSIFICATION"] == 2)
    (x, y, z) = (points['X'][ground], points['Y'][ground], points['Z'][ground])
    (row, col) = xyToRowCol(x, y, minX, maxY, pixelSize)
    nullVal = 9999.0
    minHeight = numpy.zeros((nRows, nCols), dtype=numpy.float32) + nullVal
    minHeight = minGridding(minHeight, row, col, z)
    nullArray = (minHeight == nullVal)
    xVals = pxlCoords[0][~nullArray]
    yVals = pxlCoords[1][~nullArray]
    zVals = minHeight[~nullArray]
    dem = interpGrid(xVals, yVals, zVals, pxlCoords)
    invalid = numpy.isnan(dem)
    dem[invalid] = nullVal
    dem = numpy.round(dem.astype(numpy.float32), 3)
    writeImage(dem, demFile, driver='GTiff', tlx=minX, tly=maxY,
               binsize=pixelSize, epsg=epsg, nullVal=nullVal)

    # Create canopy height model
    chm = dsm - dem
    chm[chm < 0] = 0
    chm[invalid] = nullVal
    writeImage(chm, chmFile, driver='GTiff', tlx=minX, tly=maxY,
               binsize=pixelSize, epsg=epsg, nullVal=nullVal)


def maxGridding(grid, row, col, prop):
    """
    Create grid of maximum value of prop.
    """
    numPts = len(row)
    for i in range(numPts):
        (r, c) = (row[i], col[i])
        if prop[i] > grid[r, c]:
            grid[r, c] = prop[i]
    return(grid)


def minGridding(grid, row, col, prop):
    """
    Create grid of minimum value of prop.
    """
    numPts = len(row)
    for i in range(numPts):
        (r, c) = (row[i], col[i])
        if prop[i] < grid[r, c]:
            grid[r, c] = prop[i]    
    return(grid)


def interpGrid(xVals, yVals, zVals, gridCoords):
    """
    A function to interpolate values to a regular gridCoords given 
    an irregular set of input data points
    
    Modified from pylidar/toolbox/interpolation.py
    """
    xVals = xVals.astype(numpy.float64)
    yVals = yVals.astype(numpy.float64)
    zVals = zVals.astype(numpy.float64)
    if isinstance(gridCoords, numpy.ndarray):
        gridCoords = gridCoords.astype(numpy.float64)
    else:
        gridCoords = (gridCoords[0].astype(numpy.float64),
                      gridCoords[1].astype(numpy.float64))
    
    out = pynninterp.NaturalNeighbour(xVals, yVals, zVals, gridCoords[0], gridCoords[1])

    return out

            
def writeImage(image, outfile, driver='GTiff', tlx=0.0, tly=0.0, binsize=0.0,
               epsg=None, nullVal=None):
    """
    Write data to a GDAL supported image file format
    """
    if len(image.shape)==2:
        ny,nx = image.shape
        nz=1
    if len(image.shape)==3:
        nz,ny,nx = image.shape            
    driver = gdal.GetDriverByName(driver)
    dt = image.dtype
    
    if dt == 'uint8': gdaldtype = gdal.GDT_Byte
    if dt == 'int16': gdaldtype = gdal.GDT_Int16
    if dt == 'uint16': gdaldtype = gdal.GDT_UInt16
    if dt == 'int32': gdaldtype = gdal.GDT_Int32
    if dt == 'float32': gdaldtype = gdal.GDT_Float32
    if dt == 'float64': gdaldtype = gdal.GDT_Float64
    
    ds = driver.Create(outfile, nx, ny, nz, gdaldtype, ['COMPRESS=LZW'])
    ds.SetGeoTransform([tlx,binsize,0,tly,0,-binsize])

    if epsg is not None:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(epsg)
        ds.SetProjection(proj.ExportToWkt())
    if nz>1:
        for i in range(nz):
            band = ds.GetRasterBand(i+1)
            band.WriteArray(image[i,:,:],0,0)
    else:
        band = ds.GetRasterBand(1)    
        band.WriteArray(image,0,0)
    
    # Set the null value on every band
    if nullVal is not None:
        for i in range(nz):
            band = ds.GetRasterBand(i+1)
            band.SetNoDataValue(nullVal)

    ds.FlushCache()
    

def get_uneven_grid(xr, yr, xst, yst, psize):
    """
    Builds paired x and y grid locations for interpolation with pynninterp
    
    Modified to handle floating point xst and yst
    
    """  
    x_id = (numpy.array(range(int(xr))) * psize) + xst
    x_ids = (0.5 * psize) + numpy.ones((int(yr), 1), numpy.float64) * x_id
    y_id = yst - (numpy.array(range(int(yr))) * psize)
    y_ids = numpy.ones((int(xr), 1), numpy.float64) * y_id - (0.5 * psize)
    y_ids = numpy.rot90(y_ids, 3)
    pxlCoords = [x_ids, y_ids]
    return pxlCoords

        
def xyToRowCol(x, y, xMin, yMax, pixSize):
    """
    For the given pixel size and xMin, yMax, convert the given arrays of x and y
    into arrays of row and column in a regular grid across the tile.
    
    Modified to handle floating point xMin and yMax
    
    """
    col = ((x - xMin) / pixSize).astype(numpy.uint32)
    row = ((yMax - y) / pixSize).astype(numpy.uint32)
    return (row, col)


# First copy all files to local drive
masterDir = r'S:/fowlers_gap/Sramek_EdwardsExclosures'
dirList = glob.glob(os.path.join(masterDir, '*/'))
for projectDir in dirList:
    
    # Find files
    lazFile = "None"
    mosaicImage = "None"
    lazFiles = glob.glob(os.path.join(os.path.join(projectDir, '2_densification/point_cloud'),
                         '*_group1_densified_point_cloud_classified.laz'))
    if len(lazFiles) > 0:
        lazFile = lazFiles[0]
    mosaicImages = glob.glob(os.path.join(os.path.join(projectDir, '3_dsm_ortho/2_mosaic'),
                            '*_transparent_mosaic_group1.tif'))
    if len(mosaicImages) > 0:
        mosaicImage = mosaicImages[0]
    
    if (lazFile != "None") and (mosaicImage != "None"):
        
        # Create project name as paddock_site_treatment
        project = os.path.basename(os.path.split(projectDir)[0])
        p = project.split("_")
        paddock = p[0].lower()
        if paddock[0] in ["l", "m"]:
            site = paddock[-1]
            paddock = paddock[:-1]
            treatment = p[1].lower()
        else:
            site = p[1]
            treatment = p[2].lower()
        newProject = '%s_%s_%s'%(paddock, site, treatment)
        mainDir = 'C:/Data/edwards_exclosures'
        newProjectDir = os.path.join(mainDir, newProject)
        if os.path.exists(newProjectDir) is False:
            os.mkdir(newProjectDir)
        
        newLaz = os.path.join(newProjectDir, '%s_classified.laz'%newProject)
        if os.path.exists(newLaz) is False:
            shutil.copy(lazFile, newLaz)

        mosaic_5mm = newLaz.replace('_classified.laz', '_mosaic_5mm.tif')
        if os.path.exists(mosaic_5mm) is False:
            print(newProject)
            process_laz(newLaz, mosaicImage)
