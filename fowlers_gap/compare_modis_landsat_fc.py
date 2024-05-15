#!/usr/bin/env python
"""
"""
import os, sys, glob
import numpy as np
from osgeo import ogr
from rios import applier
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
import pandas as pd
from scipy import signal


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def getModisPixelValues(info, inputs, outputs, otherargs):
    """
    """
    poly = inputs.poly[0]
    fc = inputs.fc
    polysPresent = np.unique(poly[poly != 0])
    if len(polysPresent) > 0:
    
        if len(polysPresent) > 1:
            print(len(polysPresent))
    
        uids = poly[poly != 0]
        bare = fc[0][poly != 0]
        green = fc[1][poly != 0]
        dead = fc[2][poly != 0]
        
        if len(uids) > 0:
            for i in range(uids.size):
                ID = uids[i]
                outfile = os.path.join(otherargs.outdir, 'modis_fractionalcover_%i.csv'%(ID))
                with open(outfile, 'a') as f:
                    f.write('%s,%i,%i,%i\n'%(otherargs.date, bare[i], green[i], dead[i]))


def extract_modis_fc():
    """
    Uses RIOS to extract monthly MODIS fractional cover for sample polygons.
    """
    shapefile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test\warrens_control_modis_pixel_wgs84.shp'
    baseRaster = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test\modis_fc_200101.tif'
    outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test'
    
    # Read in Id values from shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    IDs = []
    for feature in layer:
        ID = int(feature.GetField("Id"))
        IDs.append(ID)
    layer.ResetReading()
    n = len(IDs)
    
    # Iterate over ID values creating csv files to save results
    for ID in IDs:
        outfile = os.path.join(outdir, 'modis_fractionalcover_%i.csv'%(ID))
        with open(outfile, 'w') as f:
            f.write('Date,BS,PV,NPV\n')
    
    # Iterate over FC images
    for imagefile in glob.glob(r"S:\aust\modis_fractional_cover\*.tif"):
        year = imagefile.replace(r".tif", "").split(r".")[-4][1:]
        month = imagefile.replace(r".tif", "").split(r".")[-3]
        date = year + month
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        infiles.fc = imagefile
        infiles.base = baseRaster
        infiles.poly = shapefile
        otherargs.date = date
        otherargs.outdir = outdir
        controls.setReferenceImage(baseRaster)
        controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
        controls.setBurnAttribute("Id")
        applier.apply(getModisPixelValues, infiles, outfiles,
                      otherArgs=otherargs, controls=controls)


def getLandsatPixels(info, inputs, outputs, otherargs):
    """
    """
    sites = inputs.sites[0]
    for idvalue in np.unique(sites[sites != 0]):
        singlesite = (sites == idvalue)
        fc = inputs.fc
        nodataPixels = (fc[0] == 0)
        fc = np.where(fc >= 100, fc - 100, 0)
        bare = fc[0][singlesite]
        green = fc[1][singlesite]
        dead = fc[2][singlesite]
        nodata = nodataPixels[singlesite]
        bare = bare[nodata == 0]
        green = green[nodata == 0]
        dead = dead[nodata == 0]
        outfile = os.path.join(otherargs.outdir, 'landsat_fractionalcover_%i.csv'%(idvalue))
        with open(outfile, 'a') as f:
            line = '%i,%s'%(idvalue, otherargs.date)
            if bare.size > 0:
                line = '%s,%i'%(line, bare.size)
                line = '%s,%.2f,%.2f'%(line, np.mean(bare), np.std(bare))
                line = '%s,%.2f,%.2f'%(line, np.mean(green), np.std(green))
                line = '%s,%.2f,%.2f\n'%(line, np.mean(dead), np.std(dead))
            else:
                line = '%s,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999\n'%line
            f.write(line)


def extract_landsat_fc():
    """
    This sets up RIOS to extract pixel statistics for the points.
    """
    polyFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test\warrens_control_modis_pixel_albers.shp'
    outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test'
    
    # Read in Id values from shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(polyFile, 0)
    layer = dataSource.GetLayer()
    IDs = []
    for feature in layer:
        ID = int(feature.GetField("Id"))
        IDs.append(ID)
    layer.ResetReading()
    n = len(IDs)
    
    # Iterate over ID values creating csv files to save results
    for ID in IDs:
        outfile = os.path.join(outdir, 'landsat_fractionalcover_%i.csv'%(ID))
        with open(outfile, 'w') as f:
            f.write('Id,date,pixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')

    # Iterate over images and get pixel values
    imageDir = r'S:\fowlers_gap\imagery\landsat\seasonal_fractional_cover'
    imageList = glob.glob(os.path.join(imageDir, "*.tif"))
    for imagefile in imageList:
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        controls.setBurnAttribute("Id")
        controls.setReferenceImage(imagefile)
        controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
        controls.setWindowXsize(4500)
        controls.setWindowYsize(4500)
        infiles.sites = polyFile
        infiles.fc = imagefile
        otherargs.outdir = outdir
        otherargs.date = os.path.basename(imagefile).split('_')[2][1:]
        applier.apply(getLandsatPixels, infiles, outfiles,
                      otherArgs=otherargs, controls=controls)


# Get MODIS data
#extract_modis_fc()

# Get Landsat data
#extract_landsat_fc()

# Read in MODIS data
indir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test'
m_dates = []
m_green = []
m_dead = []
csvfile = os.path.join(indir, r'modis_fractionalcover_1.csv')
with open(csvfile, 'r') as f:
    f.readline()
    for line in f:
        l = line.strip().split(',')
        m_dates.append(int(l[0]))
        m_green.append(int(l[2]))
        m_dead.append(int(l[3]))
m_green = np.array(m_green)
m_dead = np.array(m_dead)
m_datetimes = np.array([datetime.date(year=int(str(d)[:4]),
                                    month=int(str(d)[4:6]), day=15) for d in m_dates])

# Read in Landsat data
l_dates = []
l_green = []
l_dead = []
csvfile = os.path.join(indir, r'landsat_fractionalcover_1.csv')
with open(csvfile, 'r') as f:
    f.readline()
    for line in f:
        l = line.strip().split(',')
        l_dates.append(int(l[1]))
        l_green.append([float(l[5]), float(l[6])])
        l_dead.append([float(l[7]), float(l[8])])
l_green = np.array(l_green)
l_dead = np.array(l_dead)
l_green = np.ma.masked_equal(l_green, 999)
l_dead = np.ma.masked_equal(l_dead, 999)
l_datetimes = np.array([datetime.date(year=int(str(d)[:4]),
                                    month=int(str(d)[4:6]), day=15) +
                                    datetime.timedelta(days=30) for d in l_dates])

# MODIS Landsat plot
fig = plt.figure(1)
fig.set_size_inches((12, 4))
ax_pv = plt.axes([0.05, 0.1, 0.93, 0.40])
ax_pv.grid(which='both', c='0.9')
ax_pv.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax_pv.xaxis.set_major_locator(mdates.YearLocator(1))
ax_pv.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_pv.plot(m_datetimes, m_green, color='k', linewidth=1)
ax_pv.set_ylabel('Photosynthetic\nvegetation cover (%)')

lower_green = l_green[:, 0] - l_green[:, 1]
lower_green[lower_green < 0] = 0
upper_green = l_green[:, 0] + l_green[:, 1]
upper_green[upper_green > 100] = 100
ax_pv.fill_between(l_datetimes, lower_green, upper_green, alpha=0.2,
                facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
ax_pv.plot(l_datetimes, l_green[:, 0], color='darkgreen', alpha=0.7, linewidth=1)

ax_npv = plt.axes([0.05, 0.55, 0.93, 0.40])
ax_npv.grid(which='both', c='0.9')
ax_npv.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax_npv.xaxis.set_major_locator(mdates.YearLocator(1))
ax_npv.xaxis.set_ticklabels([])
ax_npv.plot(m_datetimes, m_dead, color='k', linewidth=1)
ax_npv.set_ylabel('Non-photosynthetic\nvegetation cover (%)')

lower_dead = l_dead[:, 0] - l_dead[:, 1]
lower_dead[lower_dead < 0] = 0
upper_dead = l_dead[:, 0] + l_dead[:, 1]
upper_dead[upper_dead > 100] = 100
ax_npv.fill_between(l_datetimes, lower_dead, upper_dead, alpha=0.2,
                facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
ax_npv.plot(l_datetimes, l_dead[:, 0], color='saddlebrown', alpha=0.7, linewidth=1)

plt.savefig(os.path.join(indir, r'warrens_modis_landsat_fc.png'), dpi=300)

# MODIS time series decomposition plot
m_dead = pd.Series(m_dead, index=pd.date_range(np.min(m_datetimes), periods=len(m_dead), freq='M'), name="m_dead")
dead_stl = STL(m_dead, seasonal=13) 
dead_stl_result = dead_stl.fit()
m_dead_seasonal = dead_stl_result.seasonal
m_dead_trend = dead_stl_result.trend

m_green = pd.Series(m_green, index=pd.date_range(np.min(m_datetimes), periods=len(m_green), freq='M'), name="m_green")
green_stl = STL(m_green, seasonal=13) 
green_stl_result = green_stl.fit()
m_green_seasonal = green_stl_result.seasonal
m_green_trend = green_stl_result.trend

dead_peaks, dead_peak_properties = signal.find_peaks(m_dead_trend, prominence=3)
green_peaks, green_peak_properties = signal.find_peaks(m_green_trend, prominence=3)

fig = plt.figure(2)
fig.set_size_inches((9, 4))

fig.text(0.07, 0.37, '(C) Trend')
ax1 = plt.axes([0.07, 0.12, 0.92, 0.23])
ax1.grid(which='both', c='0.9')
ax1.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel('Vegetation\ncover (%)')
ax1.plot(m_datetimes, m_green_trend, color='darkgreen', linewidth=1, alpha=0.5)
ax1.plot(m_datetimes, m_dead_trend, color='saddlebrown', linewidth=1, alpha=0.5)

ax1.plot(m_datetimes[dead_peaks], m_dead_trend[dead_peaks], 'o', markersize=2, c='k')
ax1.vlines(x=m_datetimes[dead_peaks],
           ymin=m_dead_trend[dead_peaks] - dead_peak_properties["prominences"],
           ymax=m_dead_trend[dead_peaks], color = "k", linewidth=1, alpha=0.5)

ax1.plot(m_datetimes[green_peaks], m_green_trend[green_peaks], 'o', markersize=2, c='k')
ax1.vlines(x=m_datetimes[green_peaks],
           ymin=m_green_trend[green_peaks] - green_peak_properties["prominences"],
           ymax=m_green_trend[green_peaks], color = "k", linewidth=1, alpha=0.5)

# Calculate the time difference and amplitude of subsequant green-brown pulses
#for i in range(5):
#    offset = (m_datetimes[dead_peaks[i]] - m_datetimes[green_peaks[i]]).days
#    print(offset, green_peak_properties["prominences"][i], dead_peak_properties["prominences"][i])

fig.text(0.07, 0.67, '(B) Seasonal')
ax2 = plt.axes([0.07, 0.42, 0.92, 0.23])
ax2.grid(which='both', c='0.9')
ax2.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
ax2.xaxis.set_ticklabels([])
ax2.set_ylim([-7, 11])
ax2.set_yticks([-5, 0, 5, 10])
ax2.set_ylabel('Vegetation\ncover (%)')
ax2.plot(m_datetimes, m_green_seasonal, color='darkgreen', linewidth=1, alpha=0.5)
ax2.plot(m_datetimes, m_dead_seasonal, color='saddlebrown', linewidth=1, alpha=0.5)

fig.text(0.07, 0.97, '(A) Monthly timeseries')
ax3 = plt.axes([0.07, 0.72, 0.92, 0.23])
ax3.grid(which='both', c='0.9')
ax3.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax3.xaxis.set_major_locator(mdates.YearLocator(5))
ax3.xaxis.set_minor_locator(mdates.YearLocator(1))
ax3.set_ylabel('Vegetation\ncover (%)')
ax3.xaxis.set_ticklabels([])
ax3.plot(m_datetimes, m_dead, color='saddlebrown', linewidth=1, alpha=0.5)
ax3.plot(m_datetimes, m_green, color='darkgreen', linewidth=1, alpha=0.5)

axLeg = plt.axes([0, 0, 1, 0.05], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])
axLeg.plot([8, 9], [1, 1], ls='-', c='darkgreen', lw=1, alpha=0.5)
axLeg.text(10, 0.5, r'Photosynthetic vegetation')
axLeg.plot([34, 35], [1, 1], ls='-', c='saddlebrown', lw=1, alpha=0.5)
axLeg.text(36, 0.5, r'Non-photosynthetic vegetation')
axLeg.plot(62, 1.6, 'o', markersize=2, c='k')
axLeg.plot([62, 62], [1.6, 0.4], ls='-', color = "k", linewidth=1, alpha=0.5)
axLeg.text(63, 0.5, r'Peak position (dot) and amplitude (vertical line)')

plt.savefig(os.path.join(indir, r'warrens_modis_fc_decomposition.png'), dpi=300)

# Landsat time series decomposition plot
l_dead = l_dead[:, 0]
for missingIndex in np.where(np.ma.getmask(l_dead) == True)[0]:
    l_dead[missingIndex] = np.ma.mean(l_dead[missingIndex-2: missingIndex+3])

l_green = l_green[:, 0]
for missingIndex in np.where(np.ma.getmask(l_green) == True)[0]:
    l_green[missingIndex] = np.ma.mean(l_green[missingIndex-2: missingIndex+3])

l_dead = pd.Series(l_dead, index=pd.date_range(np.min(l_datetimes), periods=len(l_dead), freq='3M'), name="l_dead")
dead_stl = STL(l_dead, seasonal=13)
dead_stl_result = dead_stl.fit()
l_dead_seasonal = dead_stl_result.seasonal
l_dead_trend = dead_stl_result.trend

l_green = pd.Series(l_green, index=pd.date_range(np.min(l_datetimes), periods=len(l_green), freq='3M'), name="l_green")
green_stl = STL(l_green, seasonal=13)
green_stl_result = green_stl.fit()
l_green_seasonal = green_stl_result.seasonal
l_green_trend = green_stl_result.trend

dead_peaks, dead_peak_properties = signal.find_peaks(l_dead_trend, prominence=3)
green_peaks, green_peak_properties = signal.find_peaks(l_green_trend, prominence=3)

fig = plt.figure(3)
fig.set_size_inches((8, 4))

fig.text(0.075, 0.31, 'Trend')
ax1 = plt.axes([0.07, 0.1, 0.92, 0.25])
ax1.grid(which='both', c='0.9')
ax1.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.plot(l_datetimes, l_green_trend, color='darkgreen', linewidth=1, alpha=0.5)
ax1.plot(l_datetimes, l_dead_trend, color='saddlebrown', linewidth=1, alpha=0.5)

ax1.plot(l_datetimes[dead_peaks], l_dead_trend[dead_peaks], 'o', markersize=2, c='k')
ax1.vlines(x=l_datetimes[dead_peaks],
           ymin=l_dead_trend[dead_peaks] - dead_peak_properties["prominences"],
           ymax=l_dead_trend[dead_peaks], color = "k", linewidth=1, alpha=0.5)

ax1.plot(l_datetimes[green_peaks], l_green_trend[green_peaks], 'o', markersize=2, c='k')
ax1.vlines(x=l_datetimes[green_peaks],
           ymin=l_green_trend[green_peaks] - green_peak_properties["prominences"],
           ymax=l_green_trend[green_peaks], color = "k", linewidth=1, alpha=0.5)


fig.text(0.075, 0.61, 'Seasonal')
ax2 = plt.axes([0.07, 0.4, 0.92, 0.25])
ax2.grid(which='both', c='0.9')
ax2.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
ax2.xaxis.set_ticklabels([])
#ax2.set_ylim([-7, 11])
ax2.set_yticks([-5, 0, 5, 10])
ax2.set_ylabel('Vegetation cover (%)')
ax2.plot(l_datetimes, l_green_seasonal, color='darkgreen', linewidth=1, alpha=0.5)
ax2.plot(l_datetimes, l_dead_seasonal, color='saddlebrown', linewidth=1, alpha=0.5)

fig.text(0.075, 0.91, 'Quarterly timeseries')
ax3 = plt.axes([0.07, 0.7, 0.92, 0.25])
ax3.grid(which='both', c='0.9')
ax3.set_xlim((datetime.date(2001, month=1, day=1), datetime.date(2024, month=1, day=1)))
ax3.xaxis.set_major_locator(mdates.YearLocator(5))
ax3.xaxis.set_minor_locator(mdates.YearLocator(1))
ax3.xaxis.set_ticklabels([])
ax3.plot(l_datetimes, l_dead, color='saddlebrown', linewidth=1, alpha=0.5)
ax3.plot(l_datetimes, l_green, color='darkgreen', linewidth=1, alpha=0.5)
ax3.fill_between(l_datetimes, lower_green, upper_green, alpha=0.2,
                 facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
ax3.fill_between(l_datetimes, lower_dead, upper_dead, alpha=0.2,
                 facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')

axLeg = plt.axes([0, 0, 1, 0.01], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])
axLeg.plot([22, 23], [2, 2], ls='-', c='darkgreen', lw=2, alpha=0.5)
axLeg.text(24, 0.5, r'Photosynthetic vegetation')
axLeg.plot([57, 58], [2, 2], ls='-', c='saddlebrown', lw=2, alpha=0.5)
axLeg.text(59, 0.5, r'Non-photosynthetic vegetation')

plt.savefig(os.path.join(indir, r'warrens_landsat_fc_decomposition.png'), dpi=300)