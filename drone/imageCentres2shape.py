#!/usr/bin/env python
"""
This makes point shapefiles of the centre coordinate of drone images, which are
organised in project folders.
"""

import os
import sys
import glob
import argparse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from osgeo import ogr, osr


def make_shapefiles(mainDir):
    """
    Just needs the mainDir that contains the project dirs, which have the JPEGS.
    """

    # List of directory names with drone photos
    dirList = [os.path.basename(x) for x in
               glob.glob(os.path.join(mainDir, '*'))]

    # Get coordinates from photos
    for imageDir in dirList:
        outshape = imageDir + ".shp"
        imageDir = os.path.join(mainDir, imageDir)
        outshape = os.path.join(mainDir, outshape)
        imageList = glob.glob(os.path.join(imageDir, "*.JPG"))
        coords = []
        for i in imageList:
            image = Image.open(i)
            image.verify()
            gps_data = {}
            info = image._getexif()
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]
            (degrees, minutes, seconds) = gps_data['GPSLatitude']
            latitude = -1 * (degrees + minutes/60.0 + seconds/3600.0)
            (degrees, minutes, seconds) = gps_data['GPSLongitude']
            longitude = degrees + minutes/60.0 + seconds/3600.0
            coords.append([longitude, latitude])
        
        # Write coordinates as shapefile
        shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outshape):
            os.remove(outshape)
        outDataSource = shpDriver.CreateDataSource(outshape)
        layerName = os.path.basename(outshape).replace('.shp', '')
        outLayer = outDataSource.CreateLayer(layerName, None, ogr.wkbPoint)
        new_field = ogr.FieldDefn('photo', ogr.OFTString)
        new_field.SetWidth(50)
        outLayer.CreateField(new_field)
        new_field = ogr.FieldDefn('id', ogr.OFTInteger)
        new_field.SetWidth(5)
        outLayer.CreateField(new_field)
        featureDefn = outLayer.GetLayerDefn()
        for i in range(len(imageList)):
            x = float(coords[i][0])
            y = float(coords[i][1])
            ID = i + 1
            name = os.path.basename(imageList[i])
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.SetPoint(0, x, y)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(pt)
            outFeature.SetField("id", ID)
            outFeature.SetField("photo", name)
            outLayer.CreateFeature(outFeature)
            outFeature = None
        outDataSource = None
        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromEPSG(4326)
        spatialRef.MorphToESRI()
        with open(outshape.replace('.shp', '.prj'), 'w') as f:
            f.write(spatialRef.ExportToWkt())
        
    print("Processing completed")


def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Makes point shapefiles of the centre coordinate of "+
                         "drone images"))
    p.add_argument("-i", "--inDir", dest="inDir", default=None,
                   help=("Input directory containing project folders."))
    cmdargs = p.parse_args()
    if cmdargs.inDir is None:
        p.print_help()
        print("Must name input directory.")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    make_shapefiles(cmdargs.inDir)