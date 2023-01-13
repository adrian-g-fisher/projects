#!/usr/bin/env python

import os
import sys
import glob

srcDir = r"S:\sturt\landsat\landsat_seasonal_fractional_cover_v3"
dstDir = r"S:\sturt\landsat\scaled_landsat_seasonal_fractional_cover_v3"
for i, inimage in enumerate(glob.glob(os.path.join(srcDir, "*.tif"))):
    date = int(os.path.basename(inimage).split('_')[2].replace('m', '')[0:4])
    if date >= 2018:
        outimage = os.path.join(dstDir, os.path.basename(inimage).replace(".tif", "_scaled.tif"))
        if os.path.exists(outimage) is False:
            cmd = "gdal_translate -scale 0 100 0 254 -q %s %s"%(inimage, outimage)
            os.system(cmd)