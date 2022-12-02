#!/usr/bin/env python

import os
import sys
import glob

srcDir = r"S:\fowlers_gap\imagery\landsat\seasonal_fractional_cover"
dstDir = r"S:\fowlers_gap\imagery\landsat\scaled_seasonal_fractional_cover"
for i, inimage in enumerate(glob.glob(os.path.join(srcDir, "*.tif"))):
    outimage = os.path.join(dstDir, os.path.basename(inimage).replace(".tif", "_scaled.tif"))
    if os.path.exists(outimage) is False:
        cmd = "gdal_translate -scale 95 200 1 255 -q %s %s"%(inimage, outimage)
        os.system(cmd)