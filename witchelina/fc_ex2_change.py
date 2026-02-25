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
    fig, axs = plt.subplots(2, sharex=True)
    fig.set_size_inches((8, 5))
    name = 'WITCHEX2'
    build_date = datetime.date(2023, 2, 15)
                   
    axs[0].fill_between(landsat['Date'][landsat['site'] == name],
                            landsat['meanGreen'][landsat['site'] == name] - landsat['stdevGreen'][landsat['site'] == name],
                            landsat['meanGreen'][landsat['site'] == name] + landsat['stdevGreen'][landsat['site'] == name],
                            alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    axs[0].plot(landsat['Date'][landsat['site'] == name],
                    landsat['meanGreen'][landsat['site'] == name],
                    color='darkgreen', linewidth=1, label='PV')
        
    axs[0].fill_between(landsat['Date'][landsat['site'] == name],
                            landsat['meanDead'][landsat['site'] == name] - landsat['stdevDead'][landsat['site'] == name],
                            landsat['meanDead'][landsat['site'] == name] + landsat['stdevDead'][landsat['site'] == name],
                            alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
    axs[0].plot(landsat['Date'][landsat['site'] == name],
                    landsat['meanDead'][landsat['site'] == name],
                    color='saddlebrown', linewidth=1, label='NPV')
    axs[0].set_ylabel('Fractional cover (%)')
        
    axs[0].set_ylim((-5, 70))
            
    # Add rainfall
    rain = np.genfromtxt(r'S:\witchelina\rainfall\witchelina_rainfall.csv', names=True, delimiter=',', dtype=None)
    rainDates = []
    for i in range(rain['Year'].size):
        year = rain['Year'][i]
        month = rain['Month'][i]
        rainDates.append(datetime.date(year=year, month=month, day=15))
    rainDates = np.array(rainDates, dtype=np.datetime64)
    rain = rfn.append_fields(rain, 'Date', rainDates)
    axs[1].bar(rain['Date'], rain['Precipitation'], color='lightskyblue', width=datetime.timedelta(days=31))
    axs[1].set_ylabel('Monthly rainfall (mm)')
    
    # Set x limits
    axs[1].set_xlim([datetime.date(1988, 1, 1), datetime.date(2025, 7, 15)])
    #axs[1].set_xlim([datetime.date(2014, 7, 15), datetime.date(2025, 7, 15)])
    
    years = mdates.YearLocator(5)   # every 5 years
    #years = mdates.YearLocator(1)   # every year
    years_fmt = mdates.DateFormatter('%Y')
    axs[1].xaxis.set_major_locator(years)
    axs[1].xaxis.set_major_formatter(years_fmt)
    
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
    
    plt.savefig('ex2_fc_timeseries_1988-2025.png', dpi=300)
    #plt.savefig('ex2_fc_timeseries_2015-2025.png', dpi=300)
    plt.close(fig)


#extract_landsat_data()
plot_fc()
