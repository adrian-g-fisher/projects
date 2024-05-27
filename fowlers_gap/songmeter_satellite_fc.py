#!/usr/bin/env python
"""
Extracts monthly MODIS and seasonal Landsat fractional cover for the songmeter
sites in Warrens and Conservation.
"""
import os, sys, glob
import datetime
import numpy as np
import numpy.lib.recfunctions as rfn
from osgeo import ogr
from rios import applier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def getModisPixelValues(info, inputs, outputs, otherargs):
    """
    """
    poly = inputs.poly[0]
    fc = inputs.fc
    bare = fc[0][poly != 0]
    green = fc[1][poly != 0]
    dead = fc[2][poly != 0]
    outfile = os.path.join(otherargs.outdir, 'modis_fractionalcover.csv')
    with open(outfile, 'a') as f:
        line = '%s,%i'%(otherargs.date, bare.size)
        line = '%s,%.2f,%.2f'%(line, np.mean(bare), np.std(bare))
        line = '%s,%.2f,%.2f'%(line, np.mean(green), np.std(green))
        line = '%s,%.2f,%.2f\n'%(line, np.mean(dead), np.std(dead))
        f.write(line)


def extract_modis_fc():
    """
    Uses RIOS to extract monthly MODIS fractional cover for sample polygons.
    """
    shapefile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\songmeter_analysis\conservation_warrens_wgs84.shp'
    baseRaster = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test\modis_fc_200101.tif'
    outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\songmeter_analysis'
    
    # Iterate over ID values creating csv files to save results
    outfile = os.path.join(outdir, 'modis_fractionalcover.csv')
    with open(outfile, 'w') as f:
        f.write('date,pixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')
    
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
        controls.setWindowXsize(100)
        controls.setWindowYsize(100)
        applier.apply(getModisPixelValues, infiles, outfiles,
                      otherArgs=otherargs, controls=controls)


def getLandsatPixels(info, inputs, outputs, otherargs):
    """
    """
    poly = inputs.sites[0]
    fc = inputs.fc
    nodataPixels = (fc[0] == 0)
    fc = np.where(fc >= 100, fc - 100, 0)
    bare = fc[0][poly != 0]
    green = fc[1][poly != 0]
    dead = fc[2][poly != 0]
    nodata = nodataPixels[poly != 0]
    bare = bare[nodata == 0]
    green = green[nodata == 0]
    dead = dead[nodata == 0]
    outfile = os.path.join(otherargs.outdir, 'landsat_fractionalcover.csv')
    with open(outfile, 'a') as f:
        line = '%s'%otherargs.date
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
    This sets up RIOS to extract pixel statistics.
    """
    polyFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\songmeter_analysis\conservation_warrens_albers.shp'
    outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\songmeter_analysis'
    
    outfile = os.path.join(outdir, 'landsat_fractionalcover.csv')
    with open(outfile, 'w') as f:
        f.write('date,pixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')

    # Iterate over images and get pixel values
    imageDir = r'S:\fowlers_gap\imagery\landsat\seasonal_fractional_cover'
    imageList = glob.glob(os.path.join(imageDir, "*.tif"))
    for imagefile in imageList:
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
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

def pheno_date(d_bytes):
    s = d_bytes.decode('utf-8')
    return datetime.datetime.strptime(s, '%Y-%m-%d').date()


def make_cool_plot():
    """
    Read in Landsat, MODIS and phenocam, and make a nice plot
    """
    indir = r'C:\Users\Adrian\Documents\GitHub\fowlers_songmeter_analysis\data'
    
    landsat_csv = r'fowlersgap_landsat_conservation_warrens.csv'
    landsat = np.genfromtxt(os.path.join(indir, landsat_csv), names=True, delimiter=',')
    landsatDates = []
    for x in landsat['date']:
        year = int(str(x)[0:4])
        month = int(str(x)[4:6]) + 1
        if month == 13:
            year += 1
            month = 1
        landsatDates.append(datetime.date(year=year, month=month, day=15))
    landsatDates = np.array(landsatDates, dtype=np.datetime64)
    landsat = rfn.append_fields(landsat, 'Date', landsatDates)
    landsat["meanGreen"] = np.ma.masked_equal(landsat["meanGreen"], 999)
    landsat["stdevGreen"] = np.ma.masked_equal(landsat["stdevGreen"], 999)
    landsat["meanDead"] = np.ma.masked_equal(landsat["meanDead"], 999)
    landsat["stdevDead"] = np.ma.masked_equal(landsat["stdevDead"], 999)
    
    modis_csv = r'fowlersgap_modis_conservation_warrens.csv'
    modis = np.genfromtxt(os.path.join(indir, modis_csv), names=True, delimiter=',')
    modisDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                           month=int(str(x)[4:6]), day=15)
                                           for x in modis['date']], dtype=np.datetime64)
    modis = rfn.append_fields(modis, 'Date', modisDates)
    
    pheno1_csv = r'fowlersgap_phenocam_conservation_fenced_daily.csv'
    pheno1 = np.genfromtxt(os.path.join(indir, pheno1_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno1 = np.ma.array(pheno1, mask=pheno1["gcc"] > 900)
    
    pheno2_csv = r'fowlersgap_phenocam_warrens_1_daily.csv'
    pheno2 = np.genfromtxt(os.path.join(indir, pheno2_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno2 = np.ma.array(pheno2, mask=pheno2["gcc"] > 900)
    
    pheno3_csv = r'fowlersgap_phenocam_warrens_5_daily.csv'
    pheno3 = np.genfromtxt(os.path.join(indir, pheno3_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno3 = np.ma.array(pheno3, mask=pheno3["gcc"] > 900)
    
    pheno4_csv = r'fowlersgap_phenocam_warrens_fenced_daily.csv'
    pheno4 = np.genfromtxt(os.path.join(indir, pheno4_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno4 = np.ma.array(pheno4, mask=pheno4["gcc"] > 900)
    
    # Make 3 panel plot with Landsat, MODIS and Phenocam
    fig, axs = plt.subplots(3, sharex=True)
    fig.set_size_inches((8, 5))
    
    axs[0].fill_between(landsat['Date'],
                        landsat['meanGreen'] - landsat['stdevGreen'],
                        landsat['meanGreen'] + landsat['stdevGreen'],
                        alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    axs[0].plot(landsat['Date'], landsat['meanGreen'], color='darkgreen', linewidth=1)
    axs[0].fill_between(landsat['Date'],
                        landsat['meanDead'] - landsat['stdevDead'],
                        landsat['meanDead'] + landsat['stdevDead'],
                        alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
    axs[0].plot(landsat['Date'], landsat['meanDead'], color='saddlebrown', linewidth=1)
    axs[0].set_ylabel('Landsat\nfractional\ncover (%)')
    
    axs[1].fill_between(modis['Date'],
                        modis['meanGreen'] - modis['stdevGreen'],
                        modis['meanGreen'] + modis['stdevGreen'],
                        alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    axs[1].plot(modis['Date'], modis['meanGreen'], color='darkgreen', linewidth=1)
    axs[1].fill_between(modis['Date'],
                        modis['meanDead'] - modis['stdevDead'],
                        modis['meanDead'] + modis['stdevDead'],
                        alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
    axs[1].plot(modis['Date'], modis['meanDead'], color='saddlebrown', linewidth=1)
    axs[1].set_ylabel('MODIS\nfractional\ncover (%)')
    
    axs[2].plot(pheno1['date'], pheno1['gcc'], color='darkgreen', linewidth=1)
    axs[2].plot(pheno2['date'], pheno2['gcc'], color='darkgreen', linewidth=1)
    axs[2].plot(pheno3['date'], pheno3['gcc'], color='darkgreen', linewidth=1)
    axs[2].plot(pheno4['date'], pheno4['gcc'], color='darkgreen', linewidth=1)
    axs[2].set_xlim([datetime.date(2022, 4, 1), datetime.date(2023, 4, 1)])
    axs[2].set_ylabel('Phenocam\ngreen chromatic\ncoordinate')
    
    plt.savefig(r'satellite_phenocam.png', dpi=300)

# Get MODIS data
#extract_modis_fc()

# Get Landsat data
#extract_landsat_fc()

# Make cool plot
make_cool_plot()
