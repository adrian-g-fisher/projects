#!/usr/bin/env python
"""

This plots the fractional cover time series for plains wanderer monitoring sites.

"""


import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
                               
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)

# Read in data
dates = []
Id = []
green = []
dead = []
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\seasonal_fc_extract.csv'
with open(csvfile, 'r') as f:
    #Id,date,pixels,meanBare,stdevBare,meanGreen,stdevGreen,meanDead,stdevDead
    f.readline()
    for line in f:
        l = line.strip().split(',')
        Id.append(int(l[0]))
        dates.append(int(l[1]))
        green.append([float(l[5]), float(l[6])])
        dead.append([float(l[7]), float(l[8])])
Id = np.array(Id)
green = np.array(green)
dead = np.array(dead)
datetimes = np.array([datetime.date(year=int(str(d)[:4]),
                                    month=int(str(d)[4:6]), day=15) +
                                    datetime.timedelta(days=30) for d in dates])

# Read in site and Id values and make dictionary
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\pw_site_ids.csv'
id2site = {}
with open(csvfile, 'r') as f:
    f.readline()
    for line in f:
        ident, site = line.strip().split(',')
        id2site[int(ident)] = site

# Plot green and dead cover over time for the 17 sites separately
for i in range(1, 18):
    
    sitename = id2site[i]

    dates = datetimes[Id == i]
    inds = dates.argsort()
    green_ts = green[(Id == i), :]
    green_ts = green_ts[inds]
    dead_ts = dead[(Id == i), :]
    dead_ts = dead_ts[inds]
    dates = dates[inds]
    green_ts = np.ma.masked_equal(green_ts, 999)
    dead_ts = np.ma.masked_equal(dead_ts, 999)

    fig = plt.figure(1)
    fig.set_size_inches((8, 4))
    
    ax1 = plt.axes([0.1, 0.2, 0.85, 0.35])
    
    ax1.set_xlim((datetime.date(1987, month=1, day=1), datetime.date(2023, month=4, day=1)))
    ax1.set_xticks([datetime.date(1990, month=1, day=1),
                    datetime.date(1995, month=1, day=1),
                    datetime.date(2000, month=1, day=1),
                    datetime.date(2005, month=1, day=1),
                    datetime.date(2010, month=1, day=1),
                    datetime.date(2015, month=1, day=1),
                    datetime.date(2020, month=1, day=1)])
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.grid(which='both', c='0.9')
    ax1.set_ylim((0, 100))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.set_ylabel('                                Vegetation cover (%)', fontsize=14)   

    lower_dead = dead_ts[:, 0] - dead_ts[:, 1]
    lower_dead[lower_dead < 0] = 0
    upper_dead = dead_ts[:, 0] + dead_ts[:, 1]
    upper_dead[upper_dead > 100] = 100
    ax1.fill_between(dates, lower_dead, upper_dead, alpha=0.5, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
    ax1.plot(dates, dead_ts[:, 0], linewidth=1.0, color='saddlebrown')

    ax2 = plt.axes([0.1, 0.60, 0.85, 0.35])
    
    ax2.set_xlim((datetime.date(1987, month=1, day=1), datetime.date(2023, month=4, day=1)))
    ax2.set_xticks([datetime.date(1990, month=1, day=1),
                    datetime.date(1995, month=1, day=1),
                    datetime.date(2000, month=1, day=1),
                    datetime.date(2005, month=1, day=1),
                    datetime.date(2010, month=1, day=1),
                    datetime.date(2015, month=1, day=1),
                    datetime.date(2020, month=1, day=1)])
    ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax2.grid(which='both', c='0.9')
    ax2.set_ylim((0, 100))
    ax2.xaxis.set_ticklabels([])

    lower_green = green_ts[:, 0] - green_ts[:, 1]
    lower_green[lower_green < 0] = 0
    upper_green = green_ts[:, 0] + green_ts[:, 1]
    upper_green[upper_green > 100] = 100
    ax2.fill_between(dates, lower_green, upper_green, alpha=0.5, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    ax2.plot(dates, green_ts[:, 0], linewidth=1.0, color='darkgreen')

    # Put legend down the bottom for PV and NPV
    axLeg = plt.axes([0, 0, 1, 0.1], frameon=False)
    axLeg.set_xlim((0, 100))
    axLeg.set_ylim((0, 2))
    axLeg.set_ylim((0, 2))
    axLeg.set_xticks([])
    axLeg.set_yticks([])
    axLeg.plot([12, 13], [1.7, 1.7], ls='-', c='darkgreen', lw=10, alpha=0.5)
    axLeg.text(15, 1.4, r'Photosynthetic vegetation', fontsize=14)
    axLeg.plot([57, 58], [1.7, 1.7], ls='-', c='saddlebrown', lw=10, alpha=0.5)
    axLeg.text(60, 1.4, r'Non-photosynthetic vegetation', fontsize=14)

    plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\plains_wanderer\fc_timeseries_graphs\%i_%s_fc_timeseries.png'%(i, sitename), dpi=300)
    