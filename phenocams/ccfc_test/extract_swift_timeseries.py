#!/usr/bin/env python

import glob
import os, sys
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime


baseDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\swift_enduro_images'
outDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test'

# Make output CSV
csvFile = os.path.join(outDir, r'swift_timeseries.csv')
if os.path.exists(csvFile) is False:
    with open(csvFile, 'w') as f:
        f.write('date,rcc,gcc,bcc\n')

# Get only 1pm images
imageList = glob.glob(os.path.join(baseDir, "*.JPG"))
images = []
dates = []
for i in imageList:
    im = Image.open(i)
    exifdata = im.getexif()
    for tag_id in exifdata:
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        if tag == r'DateTime':
            dt = exifdata.get(tag_id)
            if isinstance(dt, bytes):
                dt = dt.decode()
    (d, t) = dt.split(r' ')
    d = d.replace(r':', r'')
    t = t.replace(r':', r'')[:4]
    if t == '1300':
        if d not in dates:
            images.append(i)
            dates.append(d)

# Read in each image and extract time series
for i in range(len(images)):
    
    date = dates[i]
    
    # Read in the mask image
    maskDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\swift_enduro_masks'
    maskImage = os.path.join(maskDir, r'mask_full_scene.png')
    m = Image.open(maskImage)
    maArray = np.array(m)
    
    # Read in the RGB pixels
    im = Image.open(images[i])
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
    
    # Write data to csvfile
    with open(csvFile, 'a') as f:
        f.write('%s,%.4f,%.4f,%.4f\n'%(date, rcc, gcc, bcc))