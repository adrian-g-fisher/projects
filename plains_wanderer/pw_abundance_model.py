#!/usr/bin/env python
"""

"""
import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import cuiprogress
import datetime
from scipy import interpolate


def read_model_matrix():
    """
    Columns headings are:
    previous_6M_green,field_season_dead,predictions
    """
    
    # Read in data
    csv_file = 'predictions_matrix.csv'
    model_matrix = np.genfromtxt(csv_file, delimiter=",", skip_header=1, dtype=np.float32)
    p6g = model_matrix[:, 0]
    fsd = model_matrix[:, 1]
    pwp = model_matrix[:, 2]
    
    # Interpolate new matrix with integer values
    minX = np.floor(np.min(p6g))
    maxX = np.ceil(np.max(p6g))
    minY = np.floor(np.min(fsd))
    maxY = np.ceil(np.max(fsd))
    grid_x, grid_y = np.mgrid[minX:maxX:1, minY:maxY:1]
    out = interpolate.griddata((p6g, fsd), pwp, (grid_x, grid_y), method='cubic')
    out[np.isnan(out)] = 0
    
    return(grid_x, grid_y, out)
    

def apply_pw_model(info, inputs, outputs, otherargs):
    """
    Bare is band 0 (red)
    PV is band 1   (green)
    NPV is band 2  (blue)
    nodata value is zero
    """
    field_season_dead = inputs.fieldseason[2]
    previous_1_season_green = inputs.prev1[1]
    previous_2_season_green = inputs.prev2[1]
    
    nodata = ((field_season_dead == 0)|
              (previous_1_season_green == 0)|
              (previous_2_season_green == 0)).astype(np.uint8)
    
    field_season_dead = field_season_dead - 100
    field_season_dead[field_season_dead < 0] = 0
    field_season_dead[field_season_dead > 100] = 100
    
    previous_1_season_green = previous_1_season_green - 100
    previous_1_season_green[previous_1_season_green < 0] = 0
    previous_1_season_green[previous_1_season_green > 100] = 100
    
    previous_2_season_green = previous_2_season_green - 100
    previous_2_season_green[previous_2_season_green < 0] = 0
    previous_2_season_green[previous_2_season_green > 100] = 100
    
    previous_6M_green = previous_1_season_green + previous_2_season_green
    pw_abundance = np.zeros_like(field_season_dead).astype(np.float32)    
    (x, y, z) = otherargs.matrix
    
    rows, cols = np.shape(field_season_dead)
    for r in range(rows):
        for c in range(cols):
            if (previous_6M_green[r, c] >= np.min(x) and
                previous_6M_green[r, c] <= np.max(x) and
                field_season_dead[r, c] >= np.min(y) and
                field_season_dead[r, c] <= np.max(y)):
                pw_abundance[r, c] = z[(x == previous_6M_green[r, c]) &
                                       (y == field_season_dead[r, c])][0]
    
    outputs.pw_abundance = np.array([pw_abundance]).astype(np.float32)


def predict_PW(fieldseason_image, previous_1_season_image, previous_2_season_image):
    """
    This sets up RIOS
    """
    (x, y, z) = read_model_matrix()
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    otherargs.matrix = (x, y, z)
    controls = applier.ApplierControls()
    infiles.fieldseason = fieldseason_image
    infiles.prev1 = previous_1_season_image
    infiles.prev2 = previous_2_season_image
    outfiles.pw_abundance = os.path.join(r'S:\hay_plain\pw_abundance_model',
                                         os.path.basename(fieldseason_image).replace(".tif", "_PW.tif"))
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setProgress(cuiprogress.CUIProgressBar())
    applier.apply(apply_pw_model, infiles, outfiles, otherArgs=otherargs, controls=controls)


# Get each seasonal landsat fractional cover image and apply the PW abundance model
imagedir = r"S:\hay_plain\landsat\landsat_seasonal_fractional_cover"
imageList = np.array(glob.glob(os.path.join(imagedir, r"*.tif")))
for image in imageList:
    y = int(os.path.basename(image).split("_")[2][1:5])
    m = int(os.path.basename(image).split("_")[2][5:7])
    
    # Get previous 1 season image
    if m == 3:
        start_y = y - 1
        start_m = 12
        end_y = y
        end_m = 2
    else:
        start_y = y        
        start_m = m - 3
        end_y = y
        end_m = m - 1
    prev_1s_image = os.path.join(os.path.dirname(image),
                              'lztmre_aus_m%i%02d%i%02d_dima2_subset.tif'%(start_y,
                                                                           start_m,
                                                                           end_y,
                                                                           end_m))
    # Get previous 2 season image
    if m == 3:
        start_y = y - 1
        start_m = 9
        end_y = y - 1
        end_m = 11
    if m == 6:
        start_y = y - 1
        start_m = 12
        end_y = y
        end_m = 2
    else:
        start_y = y        
        start_m = m - 3
        end_y = y
        end_m = m - 1
    prev_2s_image = os.path.join(os.path.dirname(image),
                              'lztmre_aus_m%i%02d%i%02d_dima2_subset.tif'%(start_y,
                                                                           start_m,
                                                                           end_y,
                                                                           end_m))
    
    # If previous image exists, then run model
    if os.path.exists(prev_1s_image) is True and os.path.exists(prev_2s_image) is True:
        print(os.path.basename(image))
        predict_PW(image, prev_1s_image, prev_2s_image)