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

id2name = {1: 'WITCHEX1', 2: 'WITCHEX2', 3: 'WITCHEX3',
           4: 'WITCHCON1', 5: 'WITCHCON2', 6: 'WITCHCON3'}


def getPixels(info, inputs, outputs, otherargs):
    sites = inputs.sites[0]
    for idvalue in np.unique(sites[sites != 0]):
        singlesite = (sites == idvalue)
        fc = inputs.fc
        nodataPixels = (fc[0] == 255)
        bare = fc[0][singlesite]
        green = fc[1][singlesite]
        dead = fc[2][singlesite]
        nodata = nodataPixels[singlesite]
        bare = bare[nodata == 0]
        green = green[nodata == 0]
        dead = dead[nodata == 0]
        with open(otherargs.csvfile, 'a') as f:
            line = '%i,%s,%s'%(idvalue, id2name[idvalue], otherargs.date)
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


def extract_landsat_data():
    # Create date list
    start = 198712198802
    end = 202506202508
    dateList = []
    for y1 in range(1987, 2026):
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
        f.write('Id,site,date,pixels,'+
                'meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead\n')
    
    # Iterate over list and find FC
    for date in dateList:
        print(date)
        infiles = applier.FilenameAssociations()
        infiles.sites = r'S:\witchelina\exclosures\witchelina_exclosures_wgs84.shp'
        infiles.fc = r'S:\witchelina\seasonal_fractional_cover_v3\lztmre_sa_m%i_dp1a2_subset.tif'%date
        outfiles = applier.FilenameAssociations()
        otherargs = applier.OtherInputs()
        otherargs.csvfile = csvfile
        otherargs.date = date
        controls = applier.ApplierControls()
        controls.setBurnAttribute("Id")
        controls.setWindowXsize(3000)
        controls.setWindowYsize(3000)
        applier.apply(getPixels, infiles, outfiles, otherArgs=otherargs, controls=controls)


def plot_fc():
    
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
    landsat = rfn.append_fields(landsat, 'Date', landsatDates)
    
    # Make FC plot
    fig, axs = plt.subplots(7, sharex=True)
    fig.set_size_inches((8, 8))
    names = ['WITCHCON1', 'WITCHEX1', 'WITCHCON2', 'WITCHEX2', 'WITCHCON3', 'WITCHEX3']
    name2date = {'WITCHEX1': datetime.date(2023, 2, 15), 'WITCHEX2': datetime.date(2023, 2, 15), 'WITCHEX3': datetime.date(2024, 3, 15)}
                   
    for a in range(6):
        axs[a].fill_between(landsat['Date'][landsat['site'] == names[a]],
                            landsat['meanGreen'][landsat['site'] == names[a]] - landsat['stdevGreen'][landsat['site'] == names[a]],
                            landsat['meanGreen'][landsat['site'] == names[a]] + landsat['stdevGreen'][landsat['site'] == names[a]],
                            alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
        axs[a].plot(landsat['Date'][landsat['site'] == names[a]],
                    landsat['meanGreen'][landsat['site'] == names[a]],
                    color='darkgreen', linewidth=1, label='PV')
        
        axs[a].fill_between(landsat['Date'][landsat['site'] == names[a]],
                            landsat['meanDead'][landsat['site'] == names[a]] - landsat['stdevDead'][landsat['site'] == names[a]],
                            landsat['meanDead'][landsat['site'] == names[a]] + landsat['stdevDead'][landsat['site'] == names[a]],
                            alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
        axs[a].plot(landsat['Date'][landsat['site'] == names[a]],
                    landsat['meanDead'][landsat['site'] == names[a]],
                    color='saddlebrown', linewidth=1, label='NPV')
        
        if names[a] in ['WITCHEX1', 'WITCHEX2', 'WITCHEX3']:
            build_date = name2date[names[a]]
            axs[a].axvline(build_date, color='k', linewidth=1, linestyle="--") 
        
        axs[a].set_ylim((-5, 70))
        axs[a].set_title(names[a], y=0.78, x=0.01, va='top', ha='left', fontsize=10)
        if a == 3:
            axs[a].set_ylabel('                        Fractional cover (%)')
    
    # Add rainfall
    rain = np.genfromtxt(r'S:\witchelina\rainfall\witchelina_rainfall.csv', names=True, delimiter=',', dtype=None)
    rainDates = []
    for i in range(rain['Year'].size):
        year = rain['Year'][i]
        month = rain['Month'][i]
        rainDates.append(datetime.date(year=year, month=month, day=15))
    rainDates = np.array(rainDates, dtype=np.datetime64)
    rain = rfn.append_fields(rain, 'Date', rainDates)
    axs[6].bar(rain['Date'], rain['Precipitation'], color='lightskyblue', width=datetime.timedelta(days=31))
    axs[6].set_ylabel('Monthly\nrainfall (mm)')
    
    # Set x limits
    #axs[6].set_xlim([datetime.date(1988, 1, 1), datetime.date(2025, 7, 15)])
    axs[6].set_xlim([datetime.date(2014, 7, 15), datetime.date(2025, 7, 15)])
    
    years = mdates.YearLocator(1)   # every year
    #years = mdates.YearLocator(5)   # every 5 years
    years_fmt = mdates.DateFormatter('%Y')
    axs[6].xaxis.set_major_locator(years)
    axs[6].xaxis.set_major_formatter(years_fmt)
    
    # Legend
    axLeg = plt.axes([0.07, 0.9, 1, 0.1], frameon=False)
    axLeg.set_xlim((0, 100))
    axLeg.set_ylim((0, 2))
    axLeg.set_xticks([])
    axLeg.set_yticks([])
    axLeg.plot([13.5, 14.5], [0.5, 0.5], ls='-', c='saddlebrown', lw=7, alpha=0.2)
    axLeg.plot([13.0, 15.0], [0.5, 0.5], ls='-', c='saddlebrown', lw=1)
    axLeg.text(16, 0.38, r'Non-photosynthetic vegetation', fontsize=10)
    axLeg.plot([45.5, 46.5], [0.5, 0.5], ls='-', c='darkgreen', lw=7, alpha=0.2)
    axLeg.plot([45.0, 47.0], [0.5, 0.5], ls='-', c='darkgreen', lw=1)
    axLeg.text(48, 0.38, r'Photosynthetic vegetation', fontsize=10)
    
    #plt.savefig('exclosure_fc_timeseries_1988-2025.png', dpi=300)
    plt.savefig('exclosure_fc_timeseries_2015-2025.png', dpi=300)
    plt.close(fig)


#extract_landsat_data()
plot_fc()
