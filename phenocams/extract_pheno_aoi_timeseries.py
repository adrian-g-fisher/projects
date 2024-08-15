#!/usr/bin/env python
"""
Extracts chromatic coordinate time series from phenocam images and saves as csv
files. The input directory should have separate folders for each phenocam,
stored in:
    inDir/site_name/images
    
Mask images are stored in:
    inDir/site_name/mask

Binary masks can be created in irfanview:
 - create the mask folder, copy an image, and rename mask.JPG
 - Draw a rectange for the site area.
 - Click edit "Cut - area outside of the selection"
 - On paint dialogue increase tolerance to maximum (255), click the paint
   bucket, make sure the fill colour is white, and fill the rectangle
 - The area to include should be white (pixel value = 255)
 - The area to exclude should be black (pixel value = 0)

"""

import argparse
import glob
import os, sys
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime


def main(imageDir, maskImage, csvFile):
    """
    """
    with open(csvFile, 'w') as f:
        f.write('image,date,time,rcc,gcc,bcc\n')
    
    imageList = glob.glob(os.path.join(imageDir, '*.JPG'))
    for i in imageList:
        
        # Read in the RGB pixels
        im = Image.open(i)
        imArray = np.array(im).astype(np.float32)
                
        # Get the date
        exifdata = im.getexif()
        for tag_id in exifdata:
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == r'DateTime':
                dt = exifdata.get(tag_id)
                if isinstance(dt, bytes):
                    dt = dt.decode()
        (d, t) = dt.split(r' ')
        d = d.replace(r':', r'')
        
        # Read in the mask image
        m = Image.open(maskImage)
        maArray = np.array(m)[:, :, 0] # Use the first band
                
        red = imArray[:, :, 0][maArray == 255]
        green = imArray[:, :, 1][maArray == 255]
        blue = imArray[:, :, 2][maArray == 255]
        total = red + green + blue
        nodata = (total == 0)
        total[nodata] = 1
            
        rcc = red / total
        rcc = np.mean(rcc[nodata == 0])
            
        gcc = green / total
        gcc = np.mean(gcc[nodata == 0])
            
        bcc = blue / total
        bcc = np.mean(bcc[nodata == 0])
            
        # Write data to csvfile
        with open(csvFile, 'a') as f:
            f.write('%s,%s,%s,%.4f,%.4f,%.4f\n'%(os.path.basename(i), d, t,
                                                 rcc, gcc, bcc))


def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Extracts chromatic coordinate time series from " +
                         "phenocam images"))
    p.add_argument("-i", "--inDir", dest="inDir", default=None,
                   help=("Input directory with images"))
    p.add_argument("-m", "--mask", dest="mask", default=None,
                   help=("Input mask image for area of interest"))   
    p.add_argument("-o", "--output", dest="output", default=None,
                   help=("Output csv file"))
    cmdargs = p.parse_args()
    if (cmdargs.inDir is None and cmdargs.mask is None and
        cmdargs.output is None):
        p.print_help()
        print("Must name input directories, mask image, and output CSV file")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    main(cmdargs.inDir, cmdargs.mask, cmdargs.output)