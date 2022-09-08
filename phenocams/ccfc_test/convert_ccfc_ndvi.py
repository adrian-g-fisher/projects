#!/usr/bin/env python
"""
Convert all 1pm rainbow images to NDVI
"""

import glob
import os, sys
import numpy as np
from PIL import Image


def rgb2NDVI(r, g, b):
    """
    This converts RGB values into NDVI, using the rainbow color scale of the
    ccfc.
    """
    # Limit RGB values
    if r < 12:
        r = 12
    if r > 236:
        r = 236
    if g < 14:
        g = 14
    if g > 233:
        g = 233
    if b < 12:
        b = 12
    if b > 235:
        b = 235
    
    # NDVI < 0
    if b > r:
        g_a = np.linspace(18, 208, 100)
        ndvi_a = np.linspace(-1.0, -0.01, 100)
        ndvi = ndvi_a[np.abs(g_a - g).argmin()]
    
    # 0 <= NDVI < 0.38
    elif g > r:
        r_a = np.linspace(12, 233, 38)
        ndvi_a = np.linspace(0.0, 0.37, 38)
        ndvi = ndvi_a[np.abs(r_a - r).argmin()]

    # 0.38 <= NDVI < 0.75
    elif g > b:
        g_a = np.linspace(233, 14, 37)
        ndvi_a = np.linspace(0.38, 0.74, 37)
        ndvi = ndvi_a[np.abs(g_a - g).argmin()]
    
    # 0.75 <= NDVI
    elif b >= g:
        b_a = np.linspace(12, 223, 26)
        ndvi_a = np.linspace(0.75, 1.0, 26)
        ndvi = ndvi_a[np.abs(b_a - b).argmin()]
    
    else:
        print("RGB value not valid %i %i %i"%(r, g, b))
        sys.exit()

    return(ndvi)


baseDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\ccfc_images'


for dayDir in glob.glob(os.path.join(baseDir, '*')):
    
    # Just get the 1pm images
    ndvi_image = glob.glob(os.path.join(dayDir, '*_13_00_3*.jpg'))[0]
    
    # Read in the scaled NDVI pixels, calculate NDVI, and output image
    im = Image.open(ndvi_image)
    imArray = np.array(im).astype(np.uint8)
    imArray = imArray[:1944, :, :] # Remove the NDVI scale bar
    red = imArray[:, :, 0]
    green = imArray[:, :, 1]
    blue = imArray[:, :, 2]
    ndvi = np.zeros(red.shape, dtype=np.float32)
    for i in range(red.shape[0]):
        for j in range(red.shape[1]):
            ndvi[i, j] = rgb2NDVI(red[i, j], green[i, j], blue[i, j])
    im = Image.fromarray(ndvi)
    im.save(ndvi_image.replace('.jpg', '_NDVI.tif'))
    
    print("Created %s"%os.path.basename(ndvi_image).replace('.jpg', '_NDVI.tif'))