#!/usr/bin/env python

import os
import sys
import glob
import datetime
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rios import applier
from scipy import stats, ndimage


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)

id2name = {1: 'BOOLEX1', 2: 'BOOLEX2', 3: 'BOOLEX3',
           4: 'BOOLCON1', 5: 'BOOLCON2', 6: 'BOOLCON3'}


def makeDroneNDVI(info, inputs, outputs, otherargs):
    nodata = info.getNoDataValueFor(inputs.drone)
    red = inputs.drone[otherargs.red]
    nir = inputs.drone[otherargs.nir]
    sumRedNir = np.where(red + nir == 0, 1, red + nir)
    ndvi = (nir - red) / sumRedNir
    ndvi[red + nir == 0] = 2
    ndvi[(red == nodata) | (nir == nodata)] = 2
    outputs.ndvi = np.array([ndvi]).astype(np.float32)


def make_drone_ndvi_images():
    droneList = (glob.glob(r'F:\drone_mosaics\boolcoomatta\202303\*.tif') +
                 glob.glob(r'F:\drone_mosaics\boolcoomatta\202403\*.tif'))
    for inimage in droneList:
        dstDir = os.path.dirname(inimage)
        outimage = os.path.join(dstDir, os.path.basename(inimage).replace(".tif", "_ndvi.img"))
        print('Creating %s'%os.path.basename(outimage))
        infiles = applier.FilenameAssociations()
        infiles.drone = inimage
        outfiles = applier.FilenameAssociations()
        outfiles.ndvi = outimage
        otherargs = applier.OtherInputs()
        otherargs.red = 2
        otherargs.nir = 4
        controls = applier.ApplierControls()
        controls.setStatsIgnore(2)
        controls.setCalcStats(True) 
        applier.apply(makeDroneNDVI, infiles, outfiles, otherArgs=otherargs, controls=controls)


def getPixels(info, inputs, outputs, otherargs):
    red = inputs.sr[3].astype(np.float32)
    nir = inputs.sr[4].astype(np.float32)
    sumRedNir = np.where(red + nir == 0, 1, red + nir)
    ndvi_array = (nir - red) / sumRedNir
    sites = inputs.sites[0]
    for idvalue in np.unique(sites[sites != 0]):
        singlesite = (sites == idvalue)
        fc = inputs.fc
        nodataPixels = (fc[0] == 0)
        fc = np.where(fc >= 100, fc - 100, 0)
        bare = fc[0][singlesite]
        green = fc[1][singlesite]
        dead = fc[2][singlesite]
        ndvi = ndvi_array[singlesite]
        nodata = nodataPixels[singlesite]
        bare = bare[nodata == 0]
        green = green[nodata == 0]
        dead = dead[nodata == 0]
        ndvi = ndvi[nodata == 0]
        with open(otherargs.csvfile, 'a') as f:
            line = '%i,%s,%s'%(idvalue, id2name[idvalue], otherargs.date)
            if bare.size > 0:
                line = '%s,%i'%(line, bare.size)
                line = '%s,%.2f,%.2f'%(line, np.mean(ndvi), np.std(ndvi))
                line = '%s,%.2f,%.2f'%(line, np.mean(bare), np.std(bare))
                line = '%s,%.2f,%.2f'%(line, np.mean(green), np.std(green))
                line = '%s,%.2f,%.2f\n'%(line, np.mean(dead), np.std(dead))
            else:
                line = '%s,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999'%line
                line = '%s,999,999\n'%line
            f.write(line)


def extract_landsat_data():
    # Create date list
    start = 198712198802
    end = 202406202408
    dateList = []
    for y1 in range(2022, 2025):
        for m1 in range(3, 13, 3):
            if m1 < 12:
                y2 = y1
                m2 = m1 + 2
            else:
                y2 = y1 + 1
                m2 = 2
            date = int(r'%i%02d%i%02d'%(y1, m1, y2, m2))
            if date >= start and date <= end:
                dateList.append(date)
    
    # Create CSV file
    csvfile = r'exclosure_seasonal_landsat.csv'
    with open(csvfile, 'w') as f:
        f.write('Id,site,date,pixels,meanNDVI,stdevNDVI,'+
                'meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')
    
    # Iterate over list and find FC and SR images
    for date in dateList:
        infiles = applier.FilenameAssociations()
        infiles.sites = r'S:\boolcoomata\shapefiles\boolcoomatta_exclosures_albers.shp'
        infiles.fc = r'S:\boolcoomata\seasonal_fractional_cover\lztmre_sa_m%i_dima2_subset.tif'%date
        infiles.sr = r'S:\boolcoomata\seasonal_surface_reflectance\lzolre_sa_m%i_dbia2_subset.tif'%date
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        otherargs.csvfile = csvfile
        otherargs.date = date
        controls = applier.ApplierControls()
        controls.setBurnAttribute("Id")
        controls.setWindowXsize(3000)
        controls.setWindowYsize(3000)
        applier.apply(getPixels, infiles, outfiles, otherArgs=otherargs, controls=controls)


def getDronePixels(info, inputs, outputs, otherargs):
    idvalue = np.max(inputs.sites[0])
    ndviPixels = inputs.ndvi[0][inputs.sites[0] == idvalue]
    with open(otherargs.csvfile, 'a') as f:
        line = '%i,%s,%s'%(idvalue, id2name[idvalue], otherargs.date)
        line = '%s,%i'%(line, ndviPixels.size)
        line = '%s,%.2f,%.2f\n'%(line, np.mean(ndviPixels), np.std(ndviPixels))
        f.write(line)


def extract_drone_ndvi():
    # Create CSV file
    csvfile = r'exclosure_drone_ndvi.csv'
    with open(csvfile, 'w') as f:
        f.write('Id,site,date,pixels,meanNDVI,stdevNDVI\n')
    
    # For each drone mosaic, get the mean and stdev NDVI, and the histogram
    droneList = (glob.glob(r'F:\drone_mosaics\boolcoomatta\202303\*_ndvi.img') +
                 glob.glob(r'F:\drone_mosaics\boolcoomatta\202403\*_ndvi.img'))
    
    for droneNDVI in droneList:
        date = os.path.basename(droneNDVI).split('_')[1]
        infiles = applier.FilenameAssociations()
        infiles.sites = r'S:\boolcoomata\shapefiles\boolcoomatta_exclosures_z54.shp'
        infiles.ndvi = droneNDVI
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        otherargs.csvfile = csvfile
        otherargs.date = date
        controls = applier.ApplierControls()
        controls.setBurnAttribute("Id")
        controls.setReferenceImage(droneNDVI)
        controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
        controls.setWindowXsize(18000)
        controls.setWindowYsize(18000)
        applier.apply(getDronePixels, infiles, outfiles, otherArgs=otherargs, controls=controls)


def plot_ndvi_fc():
    
    # Get landsat data
    landsat_csv = r'exclosure_seasonal_landsat.csv'
    landsat = np.genfromtxt(landsat_csv, names=True, delimiter=',', dtype=None)
    landsatDates = []
    for d in landsat['date']:
        year = int(str(d)[0:4])
        month = int(str(d)[4:6])
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        landsatDates.append(datetime.date(year=year, month=month, day=15))
    landsatDates = np.array(landsatDates, dtype=np.datetime64)
    siteNames = [x.decode('utf-8') for x in landsat['site']]
    landsat = rfn.append_fields(landsat, 'Site', siteNames)
    landsat = rfn.append_fields(landsat, 'Date', landsatDates)
    
    # Get drone data
    drone_csv = r'exclosure_drone_ndvi.csv'
    drone = np.genfromtxt(drone_csv, names=True, delimiter=',', dtype=None)
    droneDates = np.array([datetime.date(year=int(str(d)[0:4]),
                                         month=int(str(d)[4:6]),
                                         day=int(str(d)[6:]))
                                   for d in drone['date']], dtype=np.datetime64)
    siteNames = [x.decode('utf-8') for x in drone['site']]
    drone = rfn.append_fields(drone, 'Site', siteNames)
    drone = rfn.append_fields(drone, 'Date', droneDates)
    
    # Make FC plot
    fig, axs = plt.subplots(6, sharex=True)
    fig.set_size_inches((7, 5))
    names = ['BOOLCON1', 'BOOLEX1', 'BOOLCON2', 'BOOLEX2', 'BOOLCON3', 'BOOLEX3']
    for a in range(6):
        axs[a].fill_between(landsat['Date'][landsat['Site'] == names[a]],
                            landsat['meanGreen'][landsat['Site'] == names[a]] - landsat['stdevGreen'][landsat['Site'] == names[a]],
                            landsat['meanGreen'][landsat['Site'] == names[a]] + landsat['stdevGreen'][landsat['Site'] == names[a]],
                            alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
        axs[a].plot(landsat['Date'][landsat['Site'] == names[a]],
                    landsat['meanGreen'][landsat['Site'] == names[a]],
                    color='darkgreen', linewidth=1, label='PV')
        
        axs[a].fill_between(landsat['Date'][landsat['Site'] == names[a]],
                            landsat['meanDead'][landsat['Site'] == names[a]] - landsat['stdevDead'][landsat['Site'] == names[a]],
                            landsat['meanDead'][landsat['Site'] == names[a]] + landsat['stdevDead'][landsat['Site'] == names[a]],
                            alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
        axs[a].plot(landsat['Date'][landsat['Site'] == names[a]],
                    landsat['meanDead'][landsat['Site'] == names[a]],
                    color='saddlebrown', linewidth=1, label='NPV')
        axs[a].set_ylim((-5, 70))
        axs[a].set_title(names[a], y=0.79, x=0.005, va='top', ha='left', fontsize=10)
        if a == 3:
            axs[a].set_ylabel('        Fractional cover (%)')
    
    axs[5].set_xlim([datetime.date(2023, 1, 1), datetime.date(2024, 6, 1)])
    
    # Legend
    axLeg = plt.axes([0.07, 0.9, 1, 0.1], frameon=False)
    axLeg.set_xlim((0, 100))
    axLeg.set_ylim((0, 2))
    axLeg.set_xticks([])
    axLeg.set_yticks([])
    axLeg.plot([13.5, 14.5], [0.5, 0.5], ls='-', c='saddlebrown', lw=7, alpha=0.2)
    axLeg.plot([13.0, 15.0], [0.5, 0.5], ls='-', c='saddlebrown', lw=1)
    axLeg.text(16, 0.3, r'Non-photosynthetic vegetation', fontsize=10)
    axLeg.plot([45.5, 46.5], [0.5, 0.5], ls='-', c='darkgreen', lw=7, alpha=0.2)
    axLeg.plot([45.0, 47.0], [0.5, 0.5], ls='-', c='darkgreen', lw=1)
    axLeg.text(48, 0.3, r'Photosynthetic vegetation', fontsize=10)
    
    plt.savefig('exclosure_fc_timeseries.png', dpi=300)
    plt.close(fig)

    # Make NDVI plot
    fig, axs = plt.subplots(6, sharex=True)
    fig.set_size_inches((7, 5))
    for a in range(6):
        axs[a].fill_between(landsat['Date'][landsat['Site'] == names[a]],
                            landsat['meanNDVI'][landsat['Site'] == names[a]] - landsat['stdevNDVI'][landsat['Site'] == names[a]],
                            landsat['meanNDVI'][landsat['Site'] == names[a]] + landsat['stdevNDVI'][landsat['Site'] == names[a]],
                            alpha=0.2, facecolor='green', linewidth=0.0, edgecolor='green')
        axs[a].plot(landsat['Date'][landsat['Site'] == names[a]],
                    landsat['meanNDVI'][landsat['Site'] == names[a]],
                    color='green', linewidth=1, label='NDVI')
        axs[a].set_ylim((-0.1, 0.5))
        axs[a].set_title(names[a], y=0.79, x=0.005, va='top', ha='left', fontsize=10)
        if a == 3:
            axs[a].set_ylabel('        NDVI')
        
        # Add drone values
        axs[a].errorbar(drone['Date'][drone['Site'] == names[a]],
                        drone['meanNDVI'][drone['Site'] == names[a]],
                        yerr=drone['stdevNDVI'][drone['Site'] == names[a]],
                        fmt='o', color='k', ms=3)
    
    axs[5].set_xlim([datetime.date(2023, 1, 1), datetime.date(2024, 6, 1)])
    plt.savefig('exclosure_ndvi_timeseries.png', dpi=300)
    plt.close(fig)

#make_drone_ndvi_images()
#extract_landsat_data()
#extract_drone_ndvi()
plot_ndvi_fc()
