#!/usr/bin/env python
"""
conda create -n geo rios=1.4.17 numpy scipy matplotlib statsmodels
conda activate rios

Extracts MODIS fractional cover for Big Ben dune

Plots monthly FC and rainfall as a timeseries

"""

import os, sys, glob
import datetime
import numpy as np
from rios import applier
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def getPixelValues(info, inputs, outputs, otherargs):
    """
    """
    poly = inputs.poly[0]
    fc = inputs.fc.astype(np.float32)
    polysPresent = np.unique(poly[poly != 0])
    nodata = 255
    if len(polysPresent) > 0:
        uids = poly[(poly != 0) & (fc[0] != nodata)]
        bare = fc[0][(poly != 0) & (fc[0] != nodata)]
        green = fc[1][(poly != 0) & (fc[0] != nodata)]
        dead = fc[2][(poly != 0) & (fc[0] != nodata)]
        
        bare[bare < 0] = 0
        green[green < 0] = 0
        dead[dead < 0] = 0
        
        bare[bare > 100] = 100
        green[green > 100] = 100
        dead[dead > 100] = 100
        
        if len(uids) > 0:
            for i in range(uids.size):
                otherargs.pixels.append([uids[i], bare[i], green[i], dead[i]])


def extract_fc(shapefile):
    """
    Uses RIOS to extract monthly MODIS fractional cover.
    """
    outfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\sturt_drone_study\big_ben_modis_fc.csv'
    with open(outfile, 'w') as f:
        f.write('Date,Pixel_count,BS_mean,BS_std,PV_mean,PV_std,NPV_mean,NPV_std\n')
    
    # Iterate over FC images
    for imagefile in glob.glob(r"S:\aust\modis_fractional_cover\*.tif"):
        year = imagefile.replace(r".tif", "").split(r".")[-4][1:]
        month = imagefile.replace(r".tif", "").split(r".")[-3]
        date = year + month
        
        print(imagefile)
        
        infiles = applier.FilenameAssociations()
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        controls = applier.ApplierControls()
        controls.setBurnAttribute("Id")
        infiles.fc = imagefile
        infiles.poly = shapefile
        otherargs.pixels = []
        applier.apply(getPixelValues, infiles, outfiles, otherArgs=otherargs, controls=controls)
    
        # Calculate statistics on pixels within polygons
        values = np.array(otherargs.pixels).astype(np.float32)
        if values.size > 0:
            uids = np.unique(values[:, 0])
            countValues = ndimage.sum(np.ones_like(values[:, 1]), values[:, 0], uids)
            meanBare = ndimage.mean(values[:, 1], values[:, 0], uids)
            stdBare = ndimage.standard_deviation(values[:, 1], values[:, 0], uids)
            meanGreen = ndimage.mean(values[:, 2], values[:, 0], uids)
            stdGreen = ndimage.standard_deviation(values[:, 2], values[:, 0], uids)
            meanDead = ndimage.mean(values[:, 3], values[:, 0], uids)
            stdDead = ndimage.standard_deviation(values[:, 3], values[:, 0], uids)
    
            # Write to csv
            for i in range(uids.size):
                with open(outfile, "a") as f:
                    f.write('%s,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(date, countValues[i],
                                                      meanBare[i], stdBare[i],
                                                      meanGreen[i], stdGreen[i],
                                                      meanDead[i], stdDead[i]))

def plot_fc_and_rain():
    
    # Read in monthly rainfall
    csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\sturt_drone_study\rainfall\fort_grey_rainfall.csv'
    rainData = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    rainDates = []
    for i in range(rainData['Year'].size):
        rainDates.append(datetime.date(year=rainData['Year'][i], month=rainData['Month'][i], day=1))
    rainDates = np.array(rainDates)
    
    # Read in monthly FC data
    csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\sturt_drone_study\big_ben_modis_fc.csv'
    fcData = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)
    fcDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                      month=int(str(x)[4:6]), day=15)
                                      for x in fcData["Date"]])
    
    # Make plot
    outplot = csvfile.replace('.csv', '.png')
    fig = plt.figure()
    fig, axs = plt.subplots(2, sharex=True)
    fig.set_size_inches((8, 4))
    
    lower_green = fcData["PV_mean"] - fcData["PV_std"]
    lower_green[lower_green < 0] = 0
    upper_green = fcData["PV_mean"] + fcData["PV_std"]
    upper_green[upper_green > 100] = 100
    lower_dead = fcData["NPV_mean"] - fcData["NPV_std"]
    lower_dead[lower_dead < 0] = 0
    upper_dead = fcData["NPV_mean"] + fcData["NPV_std"]
    upper_dead[upper_dead > 100] = 100
    axs[0].fill_between(fcDates, lower_green, upper_green, alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    axs[0].plot(fcDates, fcData["PV_mean"], color='darkgreen', linewidth=1, alpha=0.5)
    axs[0].fill_between(fcDates, lower_dead, upper_dead, alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
    axs[0].plot(fcDates, fcData["NPV_mean"], color='saddlebrown', linewidth=1, alpha=0.5)
    axs[0].set_ylabel('Fractional cover (%)')
    axs[1].bar(rainDates, rainData['Rainfall'], color='lightskyblue', width=datetime.timedelta(days=31), align='edge')
    axs[1].set_ylabel('Monthly rainfall (mm)')
    axs[1].xaxis.set_major_locator(mdates.YearLocator(base=2))
    axs[1].xaxis.set_minor_locator(mdates.YearLocator(base=1))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[1].set_ylim((0, 250))
    axs[1].set_xlim((datetime.date(2000, month=11, day=1), datetime.date(2024, month=12, day=31)))

    axLeg = plt.axes([0.1, 0.9, 0.9, 0.05], frameon=False)
    axLeg.set_xlim((0, 100))
    axLeg.set_ylim((0, 2))
    axLeg.set_xticks([])
    axLeg.set_yticks([])
    axLeg.plot([18, 19], [1, 1], ls='-', c='darkgreen', lw=1, alpha=0.5)
    axLeg.plot([18.4, 18.6], [1, 1], ls='-', c='darkgreen', lw=5, alpha=0.2)
    axLeg.text(20, 0.5, r'Photosynthetic vegetation')
    axLeg.plot([44, 45], [1, 1], ls='-', c='saddlebrown', lw=1, alpha=0.5)
    axLeg.plot([44.4, 44.6], [1, 1], ls='-', c='saddlebrown', lw=5, alpha=0.2)
    axLeg.text(46, 0.5, r'Non-photosynthetic vegetation')
                  
    plt.savefig(outplot, dpi=300)
    plt.close()
    

#extract_fc(r'C:\Users\Adrian\OneDrive - UNSW\Documents\sturt_drone_study\big_ben.shp')
plot_fc_and_rain()