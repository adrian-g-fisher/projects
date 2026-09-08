#!/usr/bin/env python
"""

conda activate modis
      
Extracts a sample of pixels to a CSV file, including:
 - percentiles of bare, PV, NPV
 - aridity index
 - population density
 - dryland class
 - continent

Then reads the extract and makes plots

"""

import os, sys
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from osgeo import gdal, ogr, osr
from rios import applier, cuiprogress
gdal.UseExceptions()


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


# Make dictionary for continent names and IDs
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(r'S:/global/continents/World_Continents_sinusoidal.shp', 0)
layer = dataSource.GetLayer()
contID2Name = {}
for feature in layer:
    ID = int(feature.GetField("OBJECTID_1"))
    Name = feature.GetField("CONTINENT")
    contID2Name[ID] = Name
layer.ResetReading()
dataSource = None


# Make dictionary for dryland names and IDs
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(r'S:/global/Drylands_dataset_2007/drylands_UNCCD_CBD_july2014_sinusoidal.shp', 0)
layer = dataSource.GetLayer()
drylandID2Name = {}
for feature in layer:
    ID = int(feature.GetField("HIX_ZONE"))
    Name = feature.GetField("HIX_DESC")
    if ID not in drylandID2Name:
        drylandID2Name[ID] = Name
layer.ResetReading()
dataSource = None


def extractSample(info, inputs, outputs, otherargs):
    """
    Extract pixel value samples every 8 km
    """
    p05PV = inputs.p05[0]
    p05NPV = inputs.p05[1]
    p05BS = inputs.p05[2]
    p25PV = inputs.p25[0]
    p25NPV = inputs.p25[1]
    p25BS = inputs.p25[2]
    p50PV = inputs.p50[0]
    p50NPV = inputs.p50[1]
    p50BS = inputs.p50[2]
    p75PV = inputs.p75[0]
    p75NPV = inputs.p75[1]
    p75BS = inputs.p75[2]
    p95PV = inputs.p95[0]
    p95NPV = inputs.p95[1]
    p95BS = inputs.p95[2]
    aridity = inputs.aridity[0]
    salt = inputs.salt[0]
    population = inputs.population[0]
    dryland = inputs.dryland[0]
    dryland[dryland == 0] = 1
    continent = inputs.continent[0]
    
    samples = np.zeros_like(p05PV)
    samples[::16, ::16] = 1
    samples[salt == 1] = 0
    samples[p05PV == 255] = 0
    samples[continent == 0] = 0
    
    outputs.samples = np.array([samples]).astype(np.uint8)
    
    if np.max(samples) > 0:
        eastings, northings = info.getBlockCoordArrays()
        labels, num_samples = ndimage.label(samples)
        for ID in range(1, num_samples+1):
            
            easting = eastings[labels == ID][0]
            northing = northings[labels == ID][0]            
            continent_id = continent[labels == ID][0]
            continent_name = contID2Name[continent_id]
            dryland_id = dryland[labels == ID][0]
            dryland_name = drylandID2Name[dryland_id]
            
            pop = population[labels == ID][0]
            ai = aridity[labels == ID][0]
            P05PV = p05PV[labels == ID][0]
            P05NPV = p05NPV[labels == ID][0]
            P05BS = p05BS[labels == ID][0]
            P25PV = p25PV[labels == ID][0]
            P25NPV = p25NPV[labels == ID][0]
            P25BS = p25BS[labels == ID][0]
            P50PV = p50PV[labels == ID][0]
            P50NPV = p50NPV[labels == ID][0]
            P50BS = p50BS[labels == ID][0]
            P75PV = p75PV[labels == ID][0]
            P75NPV = p75NPV[labels == ID][0]
            P75BS = p75BS[labels == ID][0]
            P95PV = p95PV[labels == ID][0]
            P95NPV = p95NPV[labels == ID][0]
            P95BS = p95BS[labels == ID][0]
            
            with open(otherargs.csv, 'a') as f:
                line = '%i,%i,%s,%s'%(easting, northing, continent_name, dryland_name)
                line = '%s,%0.2f,%0.2f,%i,%i,%i'%(line, pop, ai, P05PV, P05NPV, P05BS)
                line = '%s,%i,%i,%i'%(line, P25PV, P25NPV, P25BS)
                line = '%s,%i,%i,%i'%(line, P50PV, P50NPV, P50BS)
                line = '%s,%i,%i,%i'%(line, P75PV, P75NPV, P75BS)
                line = '%s,%i,%i,%i\n'%(line, P95PV, P95NPV, P95BS)
                f.write(line)


def make_extract():
    
    outCsv = 'global_pixel_sample.csv'
    with open(outCsv, 'w') as f:
        f.write('easting,northing,continent,dryland,population,aridity,p05PV,p05NPV,p05BS,p25PV,p25NPV,p25BS,p50PV,p50NPV,p50BS,p75PV,p75NPV,p75BS,p95PV,p95NPV,p95BS\n')
    
    infiles = applier.FilenameAssociations()
    infiles.p05 = r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p05.tif'
    infiles.p25 = r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p25.tif'
    infiles.p50 = r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p50.tif'
    infiles.p75 = r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p75.tif'
    infiles.p95 = r'S:/global/modis_fractional_cover/percentiles/FC_Monthly_Medoid_v310_MCD43A4_global_200101-202606_p95.tif'
    infiles.aridity = r'S:/global/global-aridity_v3_1/Global-AI_ET0__annual_v3_1/ai_v31_yr_modis_masked.tif'
    infiles.salt = 'S:/global/GLWD_v2_0/GLWD_v2_0_combined_classes/GLWD_v2_0_saltlakes_sinusoidal_clip_fixed.tif'
    infiles.population = 'S:/global/population/gpw_v4_population_density_rev11_2020_30_sec_sinusoidal_masked.tif'
    infiles.dryland = r'S:/global/Drylands_dataset_2007/drylands_UNCCD_CBD_july2014_sinusoidal.tif'
    infiles.continent = r'S:/global/continents/World_Continents_sinusoidal.tif'
    outfiles = applier.FilenameAssociations()
    outfiles.samples = 'sample_pixels.tif'
    otherargs = applier.OtherInputs()
    otherargs.csv = outCsv
    controls = applier.ApplierControls()
    controls.setWindowXsize(256)
    controls.setWindowYsize(256)
    controls.setStatsIgnore(0)
    controls.setCalcStats(True)
    controls.setOutputDriverName("GTiff")
    controls.setFootprintType(applier.INTERSECTION)
    applier.apply(extractSample, infiles, outfiles, otherArgs=otherargs, controls=controls)


def make_plots():
    
    # Global sample n = 2330244
    
    # Read in data and remove problem values
    csv = r'C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/global_pixel_sample.csv'
    df = pd.read_csv(csv)
    df.loc[df.p50BS > 100, 'p50BS'] = 100
    df.loc[df.p50PV > 100, 'p50PV'] = 100
    df.loc[df.p50NPV > 100, 'p50NPV'] = 100
    
    # Scatter plot of aridity vs FC
    fig = plt.figure(1)
    fig.set_size_inches((6, 2))
    ax1 = plt.axes([0.1, 0.20, 0.25, 0.75])
    ax1.hist2d(df.p50BS, df.aridity, bins=100, norm=mcolors.LogNorm(), cmap='Reds')
    ax1.set_ylabel('AI')
    ax1.set_xlabel('BS')
    ax2 = plt.axes([0.4, 0.20, 0.25, 0.75])
    ax2.hist2d(df.p50PV, df.aridity, bins=100, norm=mcolors.LogNorm(), cmap='Greens')
    ax2.set_xlabel('PV')
    ax2.set_yticklabels([])
    ax3 = plt.axes([0.7, 0.20, 0.25, 0.75])
    ax3.hist2d(df.p50NPV, df.aridity, bins=100, norm=mcolors.LogNorm(), cmap='Blues')
    ax3.set_xlabel('NPV')
    ax3.set_yticklabels([])
    plt.savefig(r'C:/Users/z9803884/OneDrive - UNSW/Documents/publications/preparation/global_arid_brown_food_webs/aridity_vs_fc.png', dpi=300)
    
    
#make_extract()
make_plots()