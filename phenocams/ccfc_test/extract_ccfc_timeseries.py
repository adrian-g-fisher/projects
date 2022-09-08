#!/usr/bin/env python

import glob
import os, sys
import colormap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags, ImageColor
from datetime import datetime

maskDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\ccfc_masks'
baseDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\ccfc_images'
outDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test'

# Make output CSV
csvFile = os.path.join(outDir, r'ccfc_timeseries.csv')
if os.path.exists(csvFile) is False:
    with open(csvFile, 'w') as f:
        f.write('date,rcc,gcc,bcc,ndvi\n')


# Now get the time series
dayDirs = glob.glob(os.path.join(baseDir, '*'))
for dayDir in dayDirs:
    date = os.path.basename(dayDir).replace('_', '')
    
    # Just get the 1pm images
    rgb_image = glob.glob(os.path.join(dayDir, '*_13_00_1*.jpg'))[0]
    ndvi_image = glob.glob(os.path.join(dayDir, '*_13_00_3*.jpg'))[0]
    ndvi_image = ndvi_image.replace('.jpg', '_NDVI.tif')
    
    # Read in the RGB pixels and calculate chromatic coordinates inside the mask
    maskImage = os.path.join(maskDir, r'mask_full_scene.png')
    m = Image.open(maskImage)
    maArray = np.array(m)  
    im = Image.open(rgb_image)
    imArray = np.array(im).astype(np.float32)
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
    
    # Read in the NDVI pixels and calculate mean inside the mask
    im = Image.open(ndvi_image)
    imArray = np.array(im).astype(np.float32)
    ndvi = imArray[maArray == 255]
    ndvi = np.mean(ndvi)
    
    # Write data to csvfile
    with open(csvFile, 'a') as f:
        f.write('%s,%.4f,%.4f,%.4f,%.4f\n'%(date, rcc, gcc, bcc, ndvi))
    