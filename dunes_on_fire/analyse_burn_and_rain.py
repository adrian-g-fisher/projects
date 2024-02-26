#!/usr/bin/env python
"""

"""
import os
import sys
import glob
import datetime
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': "Arial"}
plt.rcParams.update(params)


# Get matching csv files for dune areas
burntCSVdir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2003-2016\burntarea"
burntCSVs = glob.glob(os.path.join(burntCSVdir, "*.csv"))
duneAreas = [os.path.basename(x).replace(".csv", "").replace("burntarea_", "") for x in burntCSVs]
rainCSVdir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2003-2016\rainfall"
rainCSVs = [os.path.join(rainCSVdir, "rainfall_%s.csv"%x) for x in duneAreas]

# Iterate over dune areas
for i, duneArea in enumerate(duneAreas):
    
    # Read in rainfall data
    # Columns are ID, Date, Mean_rain, Stdev_rain, Pixel_count
    rainData = np.genfromtxt(rainCSVs[i], names=True, delimiter=',')
    rainDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                        month=int(str(x)[4:6]), day=15)
                                        for x in rainData["Date"]])

    # Read in burnt area data
    # Columns are ID, Date, Burnt_area_percent, Pixel_count
    burnData = np.genfromtxt(burntCSVs[i], names=True, delimiter=',')
    burnDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                        month=int(str(x)[4:6]), day=15)
                                        for x in burnData["Date"]])

    # Create figure of monthly rainfall and burnt area
    fig = plt.figure(i)
    fig.set_size_inches((8, 2))
    fig.text(0.1, 0.9, "%s (%s)"%(" ".join(duneArea.split('_')[0:-1]), duneArea.split('_')[-1]))
    ax1 = plt.axes([0.1, 0.15, 0.8, 0.6])
    ax1.set_xlim((datetime.date(2003, month=1, day=1),
                  datetime.date(2016, month=12, day=31)))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.set_ylabel('Mean precipitation\n(mm)')  
    ax1.bar(rainDates, rainData["Mean_rain"], color='lightskyblue',
            width=datetime.timedelta(days=31))
    if np.max(rainData["Mean_rain"]) < 100.0:
        ax1.set_ylim((0.0, 100.0))
    ax2 = ax1.twinx()
    ax2.set_ylabel('Burnt area\n(%)')  
    ax2.bar(burnDates, burnData["Burnt_area_percent"], color='red', alpha=0.5,
            width=datetime.timedelta(days=31))
    if np.max(burnData["Burnt_area_percent"]) < 1.0:
        ax2.set_ylim((0.0, 1.0))

    # Histogram of mean monthly rainfall
    rainMonths = np.array([x.month for x in rainDates])
    rainMeans = np.zeros(12, dtype=np.float32)
    for m in range(1, 13):
        rainMeans[m-1] = np.mean(rainData["Mean_rain"][(rainMonths == m) &
                            (rainDates >= datetime.date(2003, month=1, day=1))])
    ax3 = plt.axes([0.5, 0.83, 0.20, 0.15])
    bars = ax3.bar(range(12), rainMeans, color='lightskyblue', width=1, align='center')
    ax3.tick_params(axis='y', length=2, pad=2, labelsize=8)
    ax3.tick_params(axis='x', length=1, pad=1)
    ax3.set_ylabel('Mean monthly\nprecipitation (mm)', fontsize=8, rotation=0, labelpad=1,
                   horizontalalignment='right', verticalalignment='center')
    ax3.set_xticks(range(12))
    ax3.set_xticklabels([])
    ax3.set_xlim([-0.5, 11.5])
    if np.max(rainMeans) < 50.0:
        ax3.set_ylim((0.0, 50.0))
    plt.setp(ax3.spines.values(), lw=0.5)
    
    # Histogram of fire frequency by month
    burnMonths = np.array([x.month for x in burnDates])
    burnMeans = np.zeros(12, dtype=np.float32)
    for m in range(1, 13):
        burnMeans[m-1] = np.mean(burnData["Burnt_area_percent"][burnMonths == m])
    ax4 = ax3.twinx()
    bars = ax4.bar(range(12), burnMeans, color='red', alpha=0.5, width=1, align='center')
    ax4.yaxis.set_label_position("right")
    ax4.tick_params(axis='y', length=2, pad=2, labelsize=8)
    ax4.set_ylabel('Mean monthly\nburnt area (%)', fontsize=8, rotation=0, labelpad=1,
                   horizontalalignment='left', verticalalignment='center')
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=8)
    ax4.set_xlim([-0.5, 11.5])
    if np.max(burnMeans) < 1.0:
        ax3.set_ylim((0.0, 1.0))
    
    outDir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2003-2016\timeseries_monthly"
    plt.savefig(os.path.join(outDir, r'%s.png'%duneArea), dpi=300)
    plt.close(fig)
    
    # Do different dunefields have distinct fire seasons?
    

    # 
    
    
    
    # Does burnt area correlate with previous rainfall?

