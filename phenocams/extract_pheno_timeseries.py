import glob
import os, sys
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime


baseDir = r'C:\Users\Adrian\Documents\fowlers_game_cameras\kangaroo_shades'

for siteDir in glob.glob(os.path.join(baseDir, '*')):
    site = os.path.basename(siteDir)
    print(site)
    
    csvFile = os.path.join(baseDir, r'%s_timeseries.csv'%site)
    if os.path.exists(csvFile) is False:
    
        with open(csvFile, 'w') as f:
            f.write('image,date,time,rcc,gcc,bcc\n')
    
        imageList = glob.glob(os.path.join(siteDir, '*'))
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
            maskDir = r'C:\Users\Adrian\Documents\fowlers_game_cameras\masks'
            maskImage = os.path.join(maskDir, r'%s_mask.JPG'%site)
            m = Image.open(maskImage)
            maArray = np.array(m)
        
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
                f.write('%s,%s,%s,%.4f,%.4f,%.4f\n'%(os.path.basename(i), d, t, rcc, gcc, bcc))