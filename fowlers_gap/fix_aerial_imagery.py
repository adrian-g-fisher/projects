#!/usr/bin/env python

import os
import sys
import glob

filelist = []
indir = r'S:\fowlers_gap\aerial_survey_2023\imagery'
folders = ['B1_tiles', 'B2_tiles', 'B3_tiles', 'sites']
for f in folders:
    i = os.path.join(indir, f)
    filelist.extend(glob.glob(os.path.join(i, '*.tif')))

for i, f in enumerate(filelist):
    cmd = 'gdal_edit.py -a_srs EPSG:7854 -a_nodata 255 %s'%f
    print(i+1, len(filelist), os.path.basename(f))
    os.system(cmd)