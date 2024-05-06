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
csvfile = r'exclosure_seasonal_fc.csv'
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
                                    month=int(str(d)[4:6]), day=15) for d in dates])

titles = ["Exclosure", "Outside"]          

# Plot green and dead cover over time for the six sites
fig = plt.figure(1)
fig.set_size_inches((8, 4))

fig.text(0.1, 0.50, 'No grazing', fontsize=14)
fig.text(0.1, 0.95, 'Grazing', fontsize=14)

rects  = [[0.1, 0.58, 0.85, 0.35],
          [0.1, 0.13, 0.85, 0.35]]
          
for i in [0, 1]:

    poly_id = i + 5

    ax = plt.axes(rects[i])
    ax.set_xlim((datetime.date(1988, month=1, day=1), datetime.date(2021, month=11, day=1)))
    ax.set_xticks([datetime.date(1990, month=1, day=1), datetime.date(2000, month=1, day=1),
                   datetime.date(2010, month=1, day=1), datetime.date(2020, month=1, day=1)])
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.grid(which='both', c='0.9')
    if i == 0:
        ax.set_xticklabels([])
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if i == 1:
        ax.set_ylabel('                                      Vegetation cover (%)', fontsize=14)   
    ax.set_ylim((0, 80))

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
    
# Put legend down the bottom for PV and NPV
axLeg = plt.axes([0, 0, 1, 0.1], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])
axLeg.plot([15.5, 16.5], [0.5, 0.5], ls='-', c='b', lw=10, alpha=0.3)
axLeg.plot([14.7, 17.3], [0.5, 0.5], ls='-', c='b', lw=1)
axLeg.text(19, 0.2, r'Non-photosynthetic vegetation', fontsize=14)
axLeg.plot([60.5, 61.5], [0.5, 0.5], ls='-', c='g', lw=10, alpha=0.3)
axLeg.plot([59.7, 62.3], [0.5, 0.5], ls='-', c='g', lw=1)
axLeg.text(64, 0.2, r'Photosynthetic vegetation', fontsize=14)

plt.savefig(r'warrens_timeseries.png', dpi=300)

# Make zoomed in plot

fig = plt.figure(2)
fig.set_size_inches((8, 4))

fig.text(0.1, 0.50, 'Not fenced (grazing)', fontsize=14)
fig.text(0.1, 0.95, 'Fenced (no grazing)', fontsize=14)

rects  = [[0.1, 0.58, 0.85, 0.35],
          [0.1, 0.13, 0.85, 0.35]]
for i in [0, 1]:

    poly_id = i + 5

    ax = plt.axes(rects[i])
    ax.set_xlim((datetime.date(2008, month=1, day=1), datetime.date(2022, month=1, day=1)))
    ax.set_xticks([datetime.date(2008, month=1, day=1),
                   datetime.date(2010, month=1, day=1),
                   datetime.date(2012, month=1, day=1),
                   datetime.date(2014, month=1, day=1),
                   datetime.date(2016, month=1, day=1),
                   datetime.date(2018, month=1, day=1),
                   datetime.date(2020, month=1, day=1),
                   datetime.date(2022, month=1, day=1)])
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.grid(which='both', c='0.9')
    if i == 0:
        ax.set_xticklabels([])
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if i == 1:
        ax.set_ylabel('                                      Vegetation cover (%)', fontsize=14)   
    ax.set_ylim((0, 80))

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
    
# Put legend down the bottom for PV and NPV
axLeg = plt.axes([0, 0, 1, 0.1], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])
axLeg.plot([15.5, 16.5], [0.5, 0.5], ls='-', c='b', lw=10, alpha=0.3)
axLeg.plot([14.7, 17.3], [0.5, 0.5], ls='-', c='b', lw=1)
axLeg.text(19, 0.2, r'Non-photosynthetic vegetation', fontsize=14)
axLeg.plot([60.5, 61.5], [0.5, 0.5], ls='-', c='g', lw=10, alpha=0.3)
axLeg.plot([59.7, 62.3], [0.5, 0.5], ls='-', c='g', lw=1)
axLeg.text(64, 0.2, r'Photosynthetic vegetation', fontsize=14)

plt.savefig(r'warrens_timeseries_zoom.png', dpi=300)


# Plot fenced and unfenced on one graph
fig = plt.figure(3)
fig.set_size_inches((8, 4))

fig.text(0.1, 0.95, 'Dead vegetation', fontsize=14)
fig.text(0.1, 0.50, 'Green vegetation', fontsize=14)

rects  = [[0.1, 0.58, 0.85, 0.35],
          [0.1, 0.13, 0.85, 0.35]]

ax1 = plt.axes(rects[0])
ax1.set_xlim((datetime.date(2008, month=1, day=1), datetime.date(2022, month=1, day=1)))
ax1.set_xticks([datetime.date(2008, month=1, day=1),
                   datetime.date(2010, month=1, day=1),
                   datetime.date(2012, month=1, day=1),
                   datetime.date(2014, month=1, day=1),
                   datetime.date(2016, month=1, day=1),
                   datetime.date(2018, month=1, day=1),
                   datetime.date(2020, month=1, day=1),
                   datetime.date(2022, month=1, day=1)])
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.grid(which='both', c='0.9')
ax1.set_xticklabels([])
ax1.set_ylim((0, 80))

ax2 = plt.axes(rects[1])
ax2.set_xlim((datetime.date(2008, month=1, day=1), datetime.date(2022, month=1, day=1)))
ax2.set_xticks([datetime.date(2008, month=1, day=1),
                   datetime.date(2010, month=1, day=1),
                   datetime.date(2012, month=1, day=1),
                   datetime.date(2014, month=1, day=1),
                   datetime.date(2016, month=1, day=1),
                   datetime.date(2018, month=1, day=1),
                   datetime.date(2020, month=1, day=1),
                   datetime.date(2022, month=1, day=1)])
ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
ax2.grid(which='both', c='0.9')
ax2.set_ylim((0, 80))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_ylabel('                                      Vegetation cover (%)', fontsize=14)   

poly_id = 5
dates = datetimes[Id == poly_id]
inds = dates.argsort()
green_ts = green[(Id == poly_id), :]
green_ts = green_ts[inds]
dead_ts = dead[(Id == poly_id), :]
dead_ts = dead_ts[inds]
dates = dates[inds]
green_ts_fenced = np.ma.masked_equal(green_ts, 999)
dead_ts_fenced = np.ma.masked_equal(dead_ts, 999)

poly_id = 6
dates = datetimes[Id == poly_id]
inds = dates.argsort()
green_ts = green[(Id == poly_id), :]
green_ts = green_ts[inds]
dead_ts = dead[(Id == poly_id), :]
dead_ts = dead_ts[inds]
dates = dates[inds]
green_ts_unfenced = np.ma.masked_equal(green_ts, 999)
dead_ts_unfenced = np.ma.masked_equal(dead_ts, 999)

lower_dead_fenced = dead_ts_fenced[:, 0] - dead_ts_fenced[:, 1]
lower_dead_fenced[lower_dead_fenced < 0] = 0
upper_dead_fenced = dead_ts_fenced[:, 0] + dead_ts_fenced[:, 1]
upper_dead_fenced[upper_dead_fenced > 100] = 100

lower_dead_unfenced = dead_ts_unfenced[:, 0] - dead_ts_unfenced[:, 1]
lower_dead_unfenced[lower_dead_unfenced < 0] = 0
upper_dead_unfenced = dead_ts_unfenced[:, 0] + dead_ts_unfenced[:, 1]
upper_dead_unfenced[upper_dead_unfenced > 100] = 100

ax1.fill_between(dates, lower_dead_fenced, upper_dead_fenced, alpha=0.2, facecolor='r', linewidth=0.0, edgecolor='r')
ax1.fill_between(dates, lower_dead_unfenced, upper_dead_unfenced, alpha=0.2, facecolor='k', linewidth=0.0, edgecolor='k')

lower_green_fenced = green_ts_fenced[:, 0] - green_ts_fenced[:, 1]
lower_green_fenced[lower_green_fenced < 0] = 0
upper_green_fenced = green_ts_fenced[:, 0] + green_ts_fenced[:, 1]
upper_green_fenced[upper_green_fenced > 100] = 100

lower_green_unfenced = green_ts_unfenced[:, 0] - green_ts_unfenced[:, 1]
lower_green_unfenced[lower_green_unfenced < 0] = 0
upper_green_unfenced = green_ts_unfenced[:, 0] + green_ts_unfenced[:, 1]
upper_green_unfenced[upper_green_unfenced > 100] = 100

ax2.fill_between(dates, lower_green_fenced, upper_green_fenced, alpha=0.2, facecolor='r', linewidth=0.0, edgecolor='r')
ax2.fill_between(dates, lower_green_unfenced, upper_green_unfenced, alpha=0.2, facecolor='k', linewidth=0.0, edgecolor='k')

# Put legend down the bottom for PV and NPV
axLeg = plt.axes([0, 0, 1, 0.1], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])
axLeg.plot([15.5, 16.5], [0.5, 0.5], ls='-', c='k', lw=10, alpha=0.3)
axLeg.text(19, 0.2, r'Not fenced (grazing)', fontsize=14)
axLeg.plot([60.5, 61.5], [0.5, 0.5], ls='-', c='r', lw=10, alpha=0.3)
axLeg.text(64, 0.2, r'Fenced (no grazing)', fontsize=14)

plt.savefig(r'warrens_comparison.png', dpi=300)

################################################################################
# One plot with 4 lines (fenced/unfenced + green/dead)
################################################################################

fig = plt.figure(4)
fig.set_size_inches((8, 4))

#fig.text(0.1, 0.95, 'Dead vegetation', fontsize=14)
#fig.text(0.1, 0.50, 'Green vegetation', fontsize=14)

rects  = [[0.1, 0.2, 0.85, 0.75]]

ax1 = plt.axes(rects[0])
ax1.set_xlim((datetime.date(2008, month=1, day=1), datetime.date(2022, month=1, day=1)))
ax1.set_xticks([datetime.date(2008, month=1, day=1),
                   datetime.date(2010, month=1, day=1),
                   datetime.date(2012, month=1, day=1),
                   datetime.date(2014, month=1, day=1),
                   datetime.date(2016, month=1, day=1),
                   datetime.date(2018, month=1, day=1),
                   datetime.date(2020, month=1, day=1),
                   datetime.date(2022, month=1, day=1)])
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.grid(which='both', c='0.9')
ax1.set_ylim((0, 80))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel('Vegetation cover (%)', fontsize=14)   

poly_id = 5
dates = datetimes[Id == poly_id]
inds = dates.argsort()
green_ts = green[(Id == poly_id), :]
green_ts = green_ts[inds]
dead_ts = dead[(Id == poly_id), :]
dead_ts = dead_ts[inds]
dates = dates[inds]
green_ts_fenced = np.ma.masked_equal(green_ts, 999)
dead_ts_fenced = np.ma.masked_equal(dead_ts, 999)

poly_id = 6
dates = datetimes[Id == poly_id]
inds = dates.argsort()
green_ts = green[(Id == poly_id), :]
green_ts = green_ts[inds]
dead_ts = dead[(Id == poly_id), :]
dead_ts = dead_ts[inds]
dates = dates[inds]
green_ts_unfenced = np.ma.masked_equal(green_ts, 999)
dead_ts_unfenced = np.ma.masked_equal(dead_ts, 999)

lower_dead_fenced = dead_ts_fenced[:, 0] - dead_ts_fenced[:, 1]
lower_dead_fenced[lower_dead_fenced < 0] = 0
upper_dead_fenced = dead_ts_fenced[:, 0] + dead_ts_fenced[:, 1]
upper_dead_fenced[upper_dead_fenced > 100] = 100

lower_dead_unfenced = dead_ts_unfenced[:, 0] - dead_ts_unfenced[:, 1]
lower_dead_unfenced[lower_dead_unfenced < 0] = 0
upper_dead_unfenced = dead_ts_unfenced[:, 0] + dead_ts_unfenced[:, 1]
upper_dead_unfenced[upper_dead_unfenced > 100] = 100

ax1.fill_between(dates, lower_dead_fenced, upper_dead_fenced, alpha=0.5, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
ax1.fill_between(dates, lower_dead_unfenced, upper_dead_unfenced, alpha=0.5, facecolor='sandybrown', linewidth=0.0, edgecolor='sandybrown')

lower_green_fenced = green_ts_fenced[:, 0] - green_ts_fenced[:, 1]
lower_green_fenced[lower_green_fenced < 0] = 0
upper_green_fenced = green_ts_fenced[:, 0] + green_ts_fenced[:, 1]
upper_green_fenced[upper_green_fenced > 100] = 100

lower_green_unfenced = green_ts_unfenced[:, 0] - green_ts_unfenced[:, 1]
lower_green_unfenced[lower_green_unfenced < 0] = 0
upper_green_unfenced = green_ts_unfenced[:, 0] + green_ts_unfenced[:, 1]
upper_green_unfenced[upper_green_unfenced > 100] = 100

ax1.fill_between(dates, lower_green_fenced, upper_green_fenced, alpha=0.5, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
ax1.fill_between(dates, lower_green_unfenced, upper_green_unfenced, alpha=0.5, facecolor='limegreen', linewidth=0.0, edgecolor='limegreen')

# Add line for fence
ax1.plot([datetime.date(year=2017, month=6, day=30),
          datetime.date(year=2017, month=6, day=30)],
          [0, 80], ls='--', c='k', lw=2)
ax1.text(datetime.date(year=2017, month=7, day=30), 75, r'Fence constructed', fontsize=14)

# Put legend down the bottom for PV and NPV
axLeg = plt.axes([0, 0, 1, 0.1], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])

axLeg.plot([12, 13], [0.6, 0.6], ls='-', c='limegreen', lw=10, alpha=0.5)
axLeg.text(15, 0.2, r'Green not fenced (grazing)', fontsize=14)

axLeg.plot([12, 13], [1.7, 1.7], ls='-', c='darkgreen', lw=10, alpha=0.5)
axLeg.text(15, 1.4, r'Green fenced (no grazing)', fontsize=14)

axLeg.plot([57, 58], [0.5, 0.5], ls='-', c='sandybrown', lw=10, alpha=0.5)
axLeg.text(60, 0.2, r'Brown not fenced (grazing)', fontsize=14)

axLeg.plot([57, 58], [1.7, 1.7], ls='-', c='saddlebrown', lw=10, alpha=0.5)
axLeg.text(60, 1.4, r'Brown fenced (no grazing)', fontsize=14)

plt.savefig(r'warrens_grazing_effect.png', dpi=300)



################################################################################
# Long term - One plot with 4 lines (fenced/unfenced + green/dead)
################################################################################

fig = plt.figure(5)
fig.set_size_inches((8, 4))

#fig.text(0.1, 0.95, 'Dead vegetation', fontsize=14)
#fig.text(0.1, 0.50, 'Green vegetation', fontsize=14)

rects  = [[0.1, 0.2, 0.85, 0.75]]

ax1 = plt.axes(rects[0])
ax1.set_xlim((datetime.date(1988, month=1, day=1), datetime.date(2022, month=1, day=1)))
ax1.set_xticks([datetime.date(1990, month=1, day=1), datetime.date(2000, month=1, day=1),
                datetime.date(2010, month=1, day=1), datetime.date(2020, month=1, day=1)])
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.set_ylim((0, 80))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel('Vegetation cover (%)', fontsize=14)   

poly_id = 5
dates = datetimes[Id == poly_id]
inds = dates.argsort()
green_ts = green[(Id == poly_id), :]
green_ts = green_ts[inds]
dead_ts = dead[(Id == poly_id), :]
dead_ts = dead_ts[inds]
dates = dates[inds]
green_ts_fenced = np.ma.masked_equal(green_ts, 999)
dead_ts_fenced = np.ma.masked_equal(dead_ts, 999)

poly_id = 6
dates = datetimes[Id == poly_id]
inds = dates.argsort()
green_ts = green[(Id == poly_id), :]
green_ts = green_ts[inds]
dead_ts = dead[(Id == poly_id), :]
dead_ts = dead_ts[inds]
dates = dates[inds]
green_ts_unfenced = np.ma.masked_equal(green_ts, 999)
dead_ts_unfenced = np.ma.masked_equal(dead_ts, 999)

lower_dead_fenced = dead_ts_fenced[:, 0] - dead_ts_fenced[:, 1]
lower_dead_fenced[lower_dead_fenced < 0] = 0
upper_dead_fenced = dead_ts_fenced[:, 0] + dead_ts_fenced[:, 1]
upper_dead_fenced[upper_dead_fenced > 100] = 100

lower_dead_unfenced = dead_ts_unfenced[:, 0] - dead_ts_unfenced[:, 1]
lower_dead_unfenced[lower_dead_unfenced < 0] = 0
upper_dead_unfenced = dead_ts_unfenced[:, 0] + dead_ts_unfenced[:, 1]
upper_dead_unfenced[upper_dead_unfenced > 100] = 100

ax1.fill_between(dates, lower_dead_fenced, upper_dead_fenced, alpha=0.9, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
#ax1.fill_between(dates, lower_dead_unfenced, upper_dead_unfenced, alpha=0.5, facecolor='sandybrown', linewidth=0.0, edgecolor='sandybrown')

lower_green_fenced = green_ts_fenced[:, 0] - green_ts_fenced[:, 1]
lower_green_fenced[lower_green_fenced < 0] = 0
upper_green_fenced = green_ts_fenced[:, 0] + green_ts_fenced[:, 1]
upper_green_fenced[upper_green_fenced > 100] = 100

lower_green_unfenced = green_ts_unfenced[:, 0] - green_ts_unfenced[:, 1]
lower_green_unfenced[lower_green_unfenced < 0] = 0
upper_green_unfenced = green_ts_unfenced[:, 0] + green_ts_unfenced[:, 1]
upper_green_unfenced[upper_green_unfenced > 100] = 100

ax1.fill_between(dates, lower_green_fenced, upper_green_fenced, alpha=0.9, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
#ax1.fill_between(dates, lower_green_unfenced, upper_green_unfenced, alpha=0.5, facecolor='limegreen', linewidth=0.0, edgecolor='limegreen')

# Add line for fence
#ax1.plot([datetime.date(year=2017, month=6, day=30),
#          datetime.date(year=2017, month=6, day=30)],
#          [0, 80], ls='--', c='k', lw=2)
#ax1.text(datetime.date(year=2017, month=7, day=30), 75, r'Fence constructed', fontsize=14)

# Put legend down the bottom for PV and NPV
axLeg = plt.axes([0, 0, 1, 0.1], frameon=False)
axLeg.set_xlim((0, 100))
axLeg.set_ylim((0, 2))
axLeg.set_ylim((0, 2))
axLeg.set_xticks([])
axLeg.set_yticks([])

#axLeg.plot([12, 13], [0.6, 0.6], ls='-', c='limegreen', lw=10, alpha=0.5)
#axLeg.text(15, 0.2, r'Green not fenced (grazing)', fontsize=14)

axLeg.plot([26, 27], [1.7, 1.7], ls='-', c='darkgreen', lw=10, alpha=0.9)
axLeg.text(29, 1.4, r'Green vegetation', fontsize=14)

#axLeg.plot([57, 58], [0.5, 0.5], ls='-', c='sandybrown', lw=10, alpha=0.5)
#axLeg.text(60, 0.2, r'Brown not fenced (grazing)', fontsize=14)

axLeg.plot([57, 58], [1.7, 1.7], ls='-', c='saddlebrown', lw=10, alpha=0.9)
axLeg.text(60, 1.4, r'Dead vegetation', fontsize=14)

plt.savefig(r'warrens_longterm_grazing_effect.png', dpi=300)
