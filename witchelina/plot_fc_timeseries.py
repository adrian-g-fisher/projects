#!/usr/bin/env python
"""

This plots the fractional cover time series for the areas of interest at
Witchelina.

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
csvfile = r'seasonal_fc_extract.csv'
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
titles = ["Area 1", "Area 2", "Area 3", "Area 4", "Area 5", "Area 6", "Area 7"]          





# Plot green and dead cover over time for the six sites
fig = plt.figure(1)
fig.set_size_inches((8, 10))
rects  = [[0.1, 0.88, 0.85, 0.11],
          [0.1, 0.75, 0.85, 0.11],
          [0.1, 0.62, 0.85, 0.11],
          [0.1, 0.49, 0.85, 0.11],
          [0.1, 0.36, 0.85, 0.11],
          [0.1, 0.23, 0.85, 0.11],
          [0.1, 0.10, 0.85, 0.11]]
for i in range(7):

    poly_id = i + 1

    ax = plt.axes(rects[i])
    ax.set_xlim((datetime.date(1988, month=1, day=1), datetime.date(2021, month=11, day=1)))
    ax.set_xticks([datetime.date(1990, month=1, day=1), datetime.date(2000, month=1, day=1),
                   datetime.date(2010, month=1, day=1), datetime.date(2020, month=1, day=1)])
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.grid(which='both', c='0.9')
    if i < 6:
        ax.set_xticklabels([])
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if i == 3:
        ax.set_ylabel('                Vegetation cover (%)', fontsize=14)   
    ax.set_ylim((0, 60))
    ax.text(datetime.date(1988, month=6, day=1), 50, titles[i], fontsize=14)

    dates = datetimes[Id == poly_id]
    inds = dates.argsort()
    green_ts = green[(Id == poly_id), :]
    green_ts = green_ts[inds]
    dead_ts = dead[(Id == poly_id), :]
    dead_ts = dead_ts[inds]
    dates = dates[inds]
    green_ts = np.ma.masked_equal(green_ts, 999)
    dead_ts = np.ma.masked_equal(dead_ts, 999)
    
    lower_green = green_ts[:, 0] - green_ts[:, 1]
    lower_green[lower_green < 0] = 0
    upper_green = green_ts[:, 0] + green_ts[:, 1]
    upper_green[upper_green > 100] = 100
    ax.fill_between(dates, lower_green, upper_green, alpha=0.2, facecolor='g', linewidth=0.0, edgecolor='g')
    ax.plot(dates, green_ts[:, 0], color='g', linewidth=1)
    
    lower_dead = dead_ts[:, 0] - dead_ts[:, 1]
    lower_dead[lower_dead < 0] = 0
    upper_dead = dead_ts[:, 0] + dead_ts[:, 1]
    upper_dead[upper_dead > 100] = 100
    ax.fill_between(dates, lower_dead, upper_dead, alpha=0.2, facecolor='b', linewidth=0.0, edgecolor='b')
    ax.plot(dates, dead_ts[:, 0], color='b', linewidth=1)
    
    print(titles[i], np.mean(green_ts[:, 0]), np.mean(dead_ts[:, 0]))
    
# Put legend down the bottom for PV and NPV
axLeg = plt.axes([0, 0, 1, 0.15], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])
axLeg.plot([15.5, 16.5], [0.5, 0.5], ls='-', c='b', lw=10, alpha=0.3)
axLeg.plot([14.7, 17.3], [0.5, 0.5], ls='-', c='b', lw=1)
axLeg.text(19, 0.4, r'Non-photosynthetic vegetation', fontsize=14)
axLeg.plot([60.5, 61.5], [0.5, 0.5], ls='-', c='g', lw=10, alpha=0.3)
axLeg.plot([59.7, 62.3], [0.5, 0.5], ls='-', c='g', lw=1)
axLeg.text(64, 0.4, r'Photosynthetic vegetation', fontsize=14)

plt.savefig(r'possible_exclosure_timeseries.png', dpi=300)