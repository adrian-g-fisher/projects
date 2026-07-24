#!/usr/bin/env python

import os
import sys
import glob
import zipfile

for zfile in glob.glob('C:/Data/murrumbidgee_2009/new_zip/*.zip'):
    with zipfile.ZipFile(zfile, 'r') as zip_ref:
        zip_ref.extractall('C:/Data/murrumbidgee_2009/new_laz')
