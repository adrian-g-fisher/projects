#!/usr/bin/env python
"""
Extracting data for analysis of grazing pressure as a function of distance to 
artifical water.
- Create distance to water raster using Landsat grid, paddock polygons, 
  paradise landsystem polygon, and AWP point locations.
- Create sample points using every fourth Landsat pixel, ignoring pixels outside
  paddock buffer, outside paradise landsystem, and outside the chenopod classified
  area.
- Extract pixel values for each sample point and create a CSV with:
    - Unique point ID number
    - Easting and northing
    - Paddock name
    - Distance to water
    - Seasonal PV/NPV/BS (198712-202411)
"""


import os
import sys
import glob
import pyproj
import numpy as np
from osgeo import ogr
from rios import applier, cuiprogress
from scipy import ndimage
ogr.UseExceptions()


def makeDistance(info, inputs, outputs, otherargs):
    """
    Iterate over 17 paddocks, calculate distance to AWP
    """
    paddocks = inputs.paddocks[0]
    awps = inputs.awps[0]
    distance = np.zeros_like(paddocks).astype(np.float32)
    for p in range(1, 18):
        print(p)
        awp = np.where(awps > 0, 0, 1)
        awp[paddocks != p] = 1
        dist_p = ndimage.distance_transform_edt(awp) * 30
        distance[paddocks == p] = dist_p[paddocks == p]
    distance[paddocks == 0] = 30000
    outputs.distance = np.array([distance])


def create_distance_raster(pointfile, polygonfile, rasterfile, distance_image):
    """
    """
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("ID")
    controls.setWindowXsize(8192)
    controls.setWindowYsize(8192)
    controls.setStatsIgnore(30000)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    infiles.awps = pointfile
    infiles.paddocks = polygonfile
    infiles.landsat = rasterfile
    outfiles.distance = distance_image
    applier.apply(makeDistance, infiles, outfiles, otherArgs=otherargs, controls=controls)


def makeSamples(info, inputs, outputs, otherargs):
    """
    """
    distance = inputs.distance[0]
    landforms = inputs.landforms[0]
    exclude = inputs.exclude[0]
    samples = np.zeros_like(landforms)
    samples[::4, ::4] = 1
    samples[distance == 30000] = 0
    samples[landforms != 1] = 0
    samples[exclude == 1] = 0
    outputs.samples = np.array([samples])
    

def create_sample_points(distance_image, landform_image, paddock_buffer, sample_image):
    """
    """
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setWindowXsize(8192)
    controls.setWindowYsize(8192)
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    infiles.distance = distance_image
    infiles.landforms = landform_image
    infiles.exclude = paddock_buffer
    outfiles.samples = sample_image
    applier.apply(makeSamples, infiles, outfiles, otherArgs=otherargs, controls=controls)


def extractValues(info, inputs, outputs, otherargs):
    """
    """
    samples = inputs.samples[0]
    if np.max(samples) > 0:
        eastings, northings = info.getBlockCoordArrays()
        labels, num_samples = ndimage.label(samples)
        for ID in range(1, num_samples+1):
            paddock_id = inputs.paddocks[0][labels == ID][0]
            paddock_name = otherargs.ID2Name[paddock_id]
            easting = eastings[labels == ID][0]
            northing = northings[labels == ID][0]
            distance = inputs.distance[0][labels == ID][0]
            
            stats = []
            images = len(inputs.fcImages)
            for i in range(images):
                seasonImage = inputs.fcImages[i]
                for b in range(3):
                    stats.append(seasonImage[b][labels == ID][0])
            
            with open(otherargs.csvfile, 'a') as f:
                line = '%i,%i,%i,%s,%0.1f'%(ID+otherargs.num,
                                              easting, northing,
                                              paddock_name, distance)
                for s in stats:
                    if s == 0:
                        s = 255
                    else:
                        s = s - 100
                        if s < 0:
                            s = 0
                        if s > 100:
                            s = 100
                    line = '%s,%.2f'%(line, s)
                f.write('%s\n'%line)
        otherargs.num += num_samples


def extract_sample_values(csvfile, sample_image, paddock_shapefile,
                          distance_image, ID2Name, seasonal_image_list):
    """
    """
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("ID")
    controls.progress = cuiprogress.CUIProgressBar()
    infiles.samples = sample_image
    infiles.distance = distance_image
    infiles.paddocks = paddock_shapefile
    infiles.fcImages = seasonal_image_list
    otherargs.num = 0
    otherargs.ID2Name = ID2Name
    otherargs.csvfile = csvfile
    applier.apply(extractValues, infiles, outfiles, otherArgs=otherargs, controls=controls)
    print("Extraction completed")


# Hardcode all the input files and directories
inDir = r'D:\witchelina\3_seasonal_analyses'
awp_shapefile = os.path.join(inDir, r'awp_epsg3577.shp')
paradise_shapefile = os.path.join(inDir, r'paradise_epsg3577.shp')
paddock_shapefile = os.path.join(inDir, r'paradise_paddocks_epsg3577.shp')
paddock_buffer = os.path.join(inDir, r'paradise_paddocks_lines_buffer100m_epsg3577.shp') # 100 m buffer
landform_image = os.path.join(inDir, r'landforms_optimum.img') # 1 = chenopods
seasonal_image_dir = r'S:\witchelina\seasonal_fractional_cover' # tif files

# Create distance to water raster
landsat_image = os.path.join(inDir, r'timeseries_stats_198712202302.tif')
distance_image = os.path.join(inDir, r'distance.tif')
#create_distance_raster(awp_shapefile, paddock_shapefile, landsat_image, distance_image)

# Create sample points
sample_image = os.path.join(inDir, r'sample_pixels.tif')
#create_sample_points(distance_image, landform_image, paddock_buffer, sample_image)

# Make dictionary for paddock names and IDs
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(paddock_shapefile, 0)
layer = dataSource.GetLayer()
ID2Name = {}
for feature in layer:
    ID = int(feature.GetField("ID"))
    Name = feature.GetField("Name")
    ID2Name[ID] = Name
layer.ResetReading()
dataSource = None

# Extract sample values to CSV
band_names = ['Bare', 'PV', 'NPV']
csvfile = os.path.join(inDir, r'awp_seasonal_analysis_epsg3577_1987_2024.csv')
header = 'ID,Easting,Northing,Paddock,Distance'

# Get seasonal dates
start = 198712198802
end = 202409202411
dateList = []
imageList = []
for y1 in range(1987, 2025):
    for m1 in range(3, 13, 3):
        if m1 < 12:
            y2 = y1
            m2 = m1 + 2
        else:
            y2 = y1 + 1
            m2 = 2
        date = int(r'%i%02d%i%02d'%(y1, m1, y2, m2))
        if date >= start and date <= end:
            dateList.append(date)
            imageList.append(os.path.join(seasonal_image_dir, r'lztmre_sa_m%i_dima2_subset.tif'%date))

for date in dateList:
    for band in band_names:
         colname = '%s_%s'%(band, str(date))
         header = '%s,%s'%(header, colname)

with open(csvfile, 'w') as f:
    f.write('%s\n'%header)

extract_sample_values(csvfile, sample_image, paddock_shapefile, distance_image, ID2Name, imageList)