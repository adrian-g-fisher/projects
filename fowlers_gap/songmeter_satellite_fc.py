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
import statsmodels.api as sm


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def getModisSurfaceRefl(info, inputs, outputs, otherargs):
    """
    """
    poly = inputs.poly[0]
    sr = inputs.sr.astype(np.float32)
    red = sr[0][poly != 0]/10000.0
    nir = sr[1][poly != 0]/10000.0
    blue = sr[2][poly != 0]/10000.0
    green = sr[3][poly != 0]/10000.0
    ndvi = (nir - red) / (nir + red)
    outfile = os.path.join(otherargs.outdir, 'modis_surface_reflectance.csv')
    with open(outfile, 'a') as f:
        line = '%s,%i'%(otherargs.date, red.size)
        line = '%s,%.2f,%.2f'%(line, np.mean(ndvi), np.std(ndvi))
        line = '%s,%.2f,%.2f,%.2f,%.2f\n'%(line, np.mean(blue), np.mean(green), np.mean(red), np.mean(nir))
        f.write(line)


def extract_modis_refl():
    """
    Uses RIOS to extract monthly MODIS surface reflectance and NDVI for sample polygons.
    """
    shapefile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\songmeter_analysis\conservation_warrens_wgs84.shp'
    baseRaster = r'C:\Users\Adrian\OneDrive - UNSW\Documents\grant_applications\2024_01_discovery_brown_food_webs\modis_test\modis_fc_200101.tif'
    outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\songmeter_analysis'
    outfile = os.path.join(outdir, 'modis_surface_reflectance.csv')
    with open(outfile, 'w') as f:
        f.write('date,pixels,meanNDVI,stdevNDVI,meanBlue,meanGreen,meanRed,meanNIR\n')

    # Iterate over images
    for imagefile in glob.glob(r"S:\aust\modis_surface_reflectance\modis_monthly_surface_reflectance\*.tif"):
        year = imagefile.replace(r".tif", "").split(r"_")[-1][0:4]
        month = imagefile.replace(r".tif", "").split(r"_")[-1][4:6]
        date = year + month
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        infiles.sr = imagefile
        infiles.base = baseRaster
        infiles.poly = shapefile
        otherargs.date = date
        otherargs.outdir = outdir
        controls.setReferenceImage(baseRaster)
        controls.setResampleMethod('cubic')
        controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
        controls.setWindowXsize(100)
        controls.setWindowYsize(100)
        applier.apply(getModisSurfaceRefl, infiles, outfiles,
                      otherArgs=otherargs, controls=controls)


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
    
    # Get Landsat data
    #landsat_csv = r'fowlersgap_landsat_conservation_warrens.csv'
    #landsat = np.genfromtxt(os.path.join(indir, landsat_csv), names=True, delimiter=',')
    #landsatDates = []
    #for x in landsat['date']:
    #    year = int(str(x)[0:4])
    #    month = int(str(x)[4:6]) + 1
    #    if month == 13:
    #        year += 1
    #        month = 1
    #    landsatDates.append(datetime.date(year=year, month=month, day=15))
    #landsatDates = np.array(landsatDates, dtype=np.datetime64)
    #landsat = rfn.append_fields(landsat, 'Date', landsatDates)
    #landsat["meanGreen"] = np.ma.masked_equal(landsat["meanGreen"], 999)
    #landsat["stdevGreen"] = np.ma.masked_equal(landsat["stdevGreen"], 999)
    #landsat["meanDead"] = np.ma.masked_equal(landsat["meanDead"], 999)
    #landsat["stdevDead"] = np.ma.masked_equal(landsat["stdevDead"], 999)
    #landsat["meanBare"] = np.ma.masked_equal(landsat["meanBare"], 999)
    #landsat["stdevBare"] = np.ma.masked_equal(landsat["stdevBare"], 999)
    
    # Get MODIS data
    modis_csv = r'fowlersgap_modis_conservation_warrens.csv'
    modis = np.genfromtxt(os.path.join(indir, modis_csv), names=True, delimiter=',')
    modisDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                           month=int(str(x)[4:6]), day=15)
                                           for x in modis['date']], dtype=np.datetime64)
    modis = rfn.append_fields(modis, 'Date', modisDates)
    
    # Get MODIS NDVI data
    modis_NDVI_csv = r'fowlersgap_modis_surface_reflectance.csv'
    modis_NDVI = np.genfromtxt(os.path.join(indir, modis_NDVI_csv), names=True, delimiter=',')
    modis_NDVI_Dates = np.array([datetime.date(year=int(str(x)[0:4]),
                                               month=int(str(x)[4:6]), day=15)
                                               for x in modis_NDVI['date']], dtype=np.datetime64)
    modis_NDVI = rfn.append_fields(modis_NDVI, 'Date', modis_NDVI_Dates)
    
    # Get Sentinel-2 NDVI data
    sentinel2_csv = r'fowlersgap_sentinel2NDVI_conservations_warrens.csv'
    sentinel2 = np.genfromtxt(os.path.join(indir, sentinel2_csv), names=True, delimiter=',')
    sentinel2Dates = np.array([datetime.date(year=int(str(x)[0:4]),
                                             month=int(str(x)[4:6]),
                                             day=int(str(x)[6:8]))
                                             for x in sentinel2['date']], dtype=np.datetime64)
    sentinel2 = rfn.append_fields(sentinel2, 'Date', sentinel2Dates)
    grass_csv = r'fowlersgap_sentinel2NDVI_grasspatch.csv'
    grass = np.genfromtxt(os.path.join(indir, grass_csv), names=True, delimiter=',')
    grassDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                         month=int(str(x)[4:6]),
                                         day=int(str(x)[6:8]))
                                         for x in grass['date']], dtype=np.datetime64)
    grass = rfn.append_fields(grass, 'Date', grassDates)
    
    # Get phenocam data
    pheno_csv = r'pheno_smooth.csv'
    phenoSmooth = np.genfromtxt(os.path.join(indir, pheno_csv), names=True, delimiter=',',
                                dtype=None, converters={0: pheno_date})
    pheno2_csv = r'fowlersgap_phenocam_warrens_1_daily.csv'
    pheno2 = np.genfromtxt(os.path.join(indir, pheno2_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno3_csv = r'fowlersgap_phenocam_warrens_5_daily.csv'
    pheno3 = np.genfromtxt(os.path.join(indir, pheno3_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno4_csv = r'fowlersgap_phenocam_warrens_fenced_daily.csv'
    pheno4 = np.genfromtxt(os.path.join(indir, pheno4_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    
    # Get rainfall data
    raindir = r'C:\Users\Adrian\Documents\GitHub\fowlers_songmeter_analysis\data\IDCJAC0009_046128_1800'
    rain_csv = r'IDCJAC0009_046128_1800_Data.csv'
    rain = np.genfromtxt(os.path.join(raindir, rain_csv), names=True, delimiter=',')
    rainDates = np.array([datetime.date(year=int(rain['Year'][i]),
                                        month=int(rain['Month'][i]),
                                        day=int(rain['Day'][i]))
                                        for i in range(rain['Year'].size)], dtype=np.datetime64)
    rain = rfn.append_fields(rain, 'Date', rainDates)
    rain["Rainfall_amount_millimetres"][np.isnan(rain["Rainfall_amount_millimetres"])] = 0
    rain = rain[rain['Date'] > datetime.date(year=2004, month=10, day=25)]
    
    # Make 5 panel plot with MODIS-FC, MODIS-NDVI, Sentinel-2-NDVI, Phenocam and rainfall
    fig, axs = plt.subplots(5, sharex=True)
    fig.set_size_inches((8, 5))
    
    axs[0].fill_between(modis['Date'],
                        modis['meanGreen'] - modis['stdevGreen'],
                        modis['meanGreen'] + modis['stdevGreen'],
                        alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    axs[0].plot(modis['Date'], modis['meanGreen'], color='darkgreen', linewidth=1,
                marker='o', markersize=2, label='PV')
    axs[0].fill_between(modis['Date'],
                        modis['meanDead'] - modis['stdevDead'],
                        modis['meanDead'] + modis['stdevDead'],
                        alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
    axs[0].plot(modis['Date'], modis['meanDead'], color='saddlebrown', linewidth=1,
                marker='o', markersize=2, label='NPV')
    #axs[0].fill_between(modis['Date'],
    #                    100 - (modis['meanBare'] - modis['stdevBare']),
    #                    100 - (modis['meanBare'] + modis['stdevBare']),
    #                    alpha=0.2, facecolor='k', linewidth=0.0, edgecolor='k')
    #axs[0].plot(modis['Date'], 100 - modis['meanBare'], color='k', linewidth=1,
    #            marker='o', markersize=2)
    axs[0].legend(loc='upper left', frameon=False, fontsize=8, labelspacing=0.2)
    axs[0].set_ylabel('MODIS\nfractional\ncover (%)')
    axs[0].set_ylim((-5, 70))
    
    axs[1].fill_between(modis_NDVI['Date'],
                        modis_NDVI['meanNDVI'] - modis_NDVI['stdevNDVI'],
                        modis_NDVI['meanNDVI'] + modis_NDVI['stdevNDVI'],
                        alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    axs[1].plot(modis_NDVI['Date'], modis_NDVI['meanNDVI'], color='darkgreen', linewidth=1,
                marker='o', markersize=2)
    axs[1].set_ylabel('MODIS\nNDVI')
    
    axs[2].fill_between(sentinel2['Date'],
                        sentinel2['NDVI_mean'] - sentinel2['NDVI_stdev'],
                        sentinel2['NDVI_mean'] + sentinel2['NDVI_stdev'],
                        alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    axs[2].plot(sentinel2['Date'], sentinel2['NDVI_mean'], color='darkgreen', linewidth=1,
                marker='o', markersize=2, label='Study area')
    axs[2].plot(grass['Date'], grass['NDVI_mean'], color='green', linewidth=1, linestyle='--', label='Grass patch')
    axs[2].legend(loc='upper left', frameon=False, fontsize=8, labelspacing=0.2)
    axs[2].set_ylabel('Sentinel-2\nNDVI')
    
    axs[3].plot(pheno2['date'], pheno2['gcc'], color='0.8', linewidth=0, marker='o', markersize=2)
    axs[3].plot(pheno3['date'], pheno3['gcc'], color='0.8', linewidth=0, marker='o', markersize=2)
    axs[3].plot(pheno4['date'], pheno4['gcc'], color='0.8', linewidth=0, marker='o', markersize=2)
    axs[3].plot(phenoSmooth['date'], phenoSmooth['gcc'], color='green', linewidth=1)
    axs[3].fill_between(phenoSmooth['date'], phenoSmooth['p05GCC'], phenoSmooth['p95GCC'], alpha=0.2, color="green", linewidth=0)
    axs[3].set_ylabel('Phenocam\ngreen\nchromatic\ncoordinate')

    axs[4].bar(rain['Date'], rain['Rainfall_amount_millimetres'],
               color='blue', width=1)
    axs[4].set_ylim((0, 60))
    axs[4].set_ylabel('Daily\nprecipitation\n(mm)')
    
    # Legend
    #axLeg = plt.axes([0, 0.9, 1, 0.1], frameon=False)
    #axLeg.set_xlim((0, 100))
    #axLeg.set_ylim((0, 2))
    #axLeg.set_xticks([])
    #axLeg.set_yticks([])
    #axLeg.plot([13.5, 14.5], [0.5, 0.5], ls='-', c='saddlebrown', lw=7, alpha=0.2)
    #axLeg.plot([13.0, 15.0], [0.5, 0.5], ls='-', c='saddlebrown', lw=1)
    #axLeg.text(16, 0.3, r'Non-photosynthetic vegetation', fontsize=10)
    #axLeg.plot([45.5, 46.5], [0.5, 0.5], ls='-', c='darkgreen', lw=7, alpha=0.2)
    #axLeg.plot([45.0, 47.0], [0.5, 0.5], ls='-', c='darkgreen', lw=1)
    #axLeg.text(48, 0.3, r'Photosynthetic vegetation', fontsize=10)
    #axLeg.plot([74.5, 75.5], [0.5, 0.5], ls='-', c='k', lw=7, alpha=0.2)
    #axLeg.plot([74.0, 76.0], [0.5, 0.5], ls='-', c='k', lw=1)
    #axLeg.text(77, 0.3, r'Total vegetation', fontsize=10)
    
    axs[4].set_xlim([datetime.date(2022, 3, 1), datetime.date(2023, 4, 1)])
    plt.savefig(r'satellite_phenocam_2022-2023.png', dpi=300)


def lowess_with_confidence_bounds(x, y, eval_x):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    N = 200
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]
        smoothed_values[i] = sm.nonparametric.lowess(exog=sampled_x, endog=sampled_y, xvals=eval_x, frac=0.08)

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - 0.95) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return bottom, top


def fit_pheno_curve():
    """
    Use lowess to fit curve
    """
    
    # Read in data
    indir = r'C:\Users\Adrian\Documents\GitHub\fowlers_songmeter_analysis\data'
    pheno2_csv = r'fowlersgap_phenocam_warrens_1_daily.csv'
    pheno2 = np.genfromtxt(os.path.join(indir, pheno2_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno3_csv = r'fowlersgap_phenocam_warrens_5_daily.csv'
    pheno3 = np.genfromtxt(os.path.join(indir, pheno3_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    pheno4_csv = r'fowlersgap_phenocam_warrens_fenced_daily.csv'
    pheno4 = np.genfromtxt(os.path.join(indir, pheno4_csv), names=True, delimiter=',',
                           dtype=None, converters={0: pheno_date})
    
    # Concatentate data
    y_values = np.concatenate((pheno2['gcc'], pheno3['gcc'], pheno4['gcc']))
    x_dates = np.concatenate((pheno2['date'], pheno3['date'], pheno4['date']))
    minDate = min(x_dates)
    x_values = np.array([(x - minDate).days for x in x_dates])
    eval_x = np.linspace(0, max(x_values), max(x_values)+1)
    eval_dates = [minDate + datetime.timedelta(days=x) for x in eval_x]
    
    # Fit lowess
    lowess = sm.nonparametric.lowess
    smooth = lowess(exog=x_values, endog=y_values, xvals=eval_x, frac=0.08)
    
    # Get 95% CI
    bottom, top = lowess_with_confidence_bounds(x_values, y_values, eval_x)
    
    # Make a plot
    fig = plt.figure()
    fig.set_size_inches((10, 4))
    ax = plt.axes([0.15, 0.10, 0.80, 0.80])
    ax.grid(which='major', axis='x', c='0.9')
    ax.set_ylabel('GCC')
    ax.plot(x_dates, y_values, color='0.8', marker='o', linewidth=0, markersize=2)
    ax.plot(eval_dates, smooth, linewidth=1, c="green")
    ax.fill_between(eval_dates, bottom, top, alpha=0.2, color="green", linewidth=0)
    plt.savefig('pheno_smooth.png', dpi=300)
    
    # Write to CSV
    with open('pheno_smooth.csv', 'w') as f:
        f.write('date,gcc,p05GCC,p95GCC\n')
        for i in range(smooth.size):
            f.write('%s,%.4f,%.4f,%.4f\n'%(eval_dates[i].strftime('%Y-%m-%d'), smooth[i], bottom[i], top[i]))
    

# Get MODIS FC data
#extract_modis_fc()

# Get Landsat FC data
#extract_landsat_fc()

# Get MODIS surface reflectance
#extract_modis_refl()

# Fit one curve to all 3 phenocam timeseries
#fit_pheno_curve()

# Make cool plot
make_cool_plot()

