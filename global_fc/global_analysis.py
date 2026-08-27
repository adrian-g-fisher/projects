#!/usr/bin/env python
"""

conda activate modis
      

Extract all pixels to CSV file, including:
 - percentiles of bare, PV, NPV
 - aridity index
 - dryland class
 - continent


      
"""
import os, sys
import numpy as np
import glob
import xarray as xr
import rioxarray
from osgeo import gdal, ogr, osr
from rios import applier, cuiprogress
gdal.UseExceptions()




