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
burntCSVdir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\burntarea"
burntCSVs = glob.glob(os.path.join(burntCSVdir, "*.csv"))
duneAreas = [os.path.basename(x).replace(".csv", "").replace("burntarea_", "") for x in burntCSVs]
rainCSVdir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\rainfall"
rainCSVs = [os.path.join(rainCSVdir, "rainfall_%s.csv"%x) for x in duneAreas]
fcCSVdir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\fractionalcover"
fcCSVs = [os.path.join(fcCSVdir, "fractionalcover_%s.csv"%x) for x in duneAreas]

# Iterate over dune areas
nameList = []
areaList = []
climateList = []
fireSeasonList = []
rainSeasonList = []
pvSeasonList = []
npvSeasonList = []
for i, duneArea in enumerate(duneAreas):
    
    nameList.append(duneArea)
    
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
    
    # Read in fractional cover data
    # Columns are Date, ID, Pixel_count, BS_mean, BS_std, PV_mean, PV_std, NPV_mean, NPV_std
    fcData = np.genfromtxt(fcCSVs[i], names=True, delimiter=',')
    fcDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                      month=int(str(x)[4:6]), day=15)
                                      for x in fcData["Date"]])
    
    # Create figure of monthly rainfall, burnt area, PV, and NPV
    fig = plt.figure(i)
    fig.set_size_inches((8, 2))
    area = " ".join(duneArea.split('_')[0:-1])
    if area == 'GSD':
        area = 'Great Sandy Desert'
    if area == 'GVD':
        area = 'Great Victoria Desert'
    areaList.append(area)
    climate = duneArea.split('_')[-1]
    climateList.append(climate)
    fig.text(0.07, 0.9, "%s (%s)"%(area, climate))
    ax1 = plt.axes([0.07, 0.15, 0.78, 0.6])
    ax1.set_xlim((datetime.date(2000, month=11, day=1),
                  datetime.date(2022, month=12, day=31)))

    # Bar plot of monthly rainfall
    ax1.set_ylabel('Precipitation (mm)')  
    ax1.bar(rainDates, rainData["Mean_rain"], color='lightskyblue',
            width=datetime.timedelta(days=31))
    if np.max(rainData["Mean_rain"]) < 100.0:
        ax1.set_ylim((0.0, 100.0))
    
    # Bar plot of monthly burnt area
    ax2 = ax1.twinx()
    ax2.set_ylabel('Burnt area (%)')  
    ax2.bar(burnDates, burnData["Burnt_area_percent"], color='red', alpha=0.5,
            width=datetime.timedelta(days=31))
    if np.max(burnData["Burnt_area_percent"]) < 1.2:
        ax2.set_ylim((0.0, 1.2))

    # Line plot of monthly PV and NPV
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.09))
    ax3.set_ylabel('Vegetation cover (%)')
    lower_green = fcData["PV_mean"] - fcData["PV_std"]
    lower_green[lower_green < 0] = 0
    upper_green = fcData["PV_mean"] + fcData["PV_std"]
    upper_green[upper_green > 100] = 100
    ax3.fill_between(fcDates, lower_green, upper_green, alpha=0.2, facecolor='darkgreen', linewidth=0.0, edgecolor='darkgreen')
    ax3.plot(fcDates, fcData["PV_mean"], color='darkgreen', linewidth=1, alpha=0.5)
    lower_dead = fcData["NPV_mean"] - fcData["NPV_std"]
    lower_dead[lower_dead < 0] = 0
    upper_dead = fcData["NPV_mean"] + fcData["NPV_std"]
    upper_dead[upper_dead > 100] = 100
    ax3.fill_between(fcDates, lower_dead, upper_dead, alpha=0.2, facecolor='saddlebrown', linewidth=0.0, edgecolor='saddlebrown')
    ax3.plot(fcDates, fcData["NPV_mean"], color='saddlebrown', linewidth=1, alpha=0.5)
    ax3.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax3.xaxis.set_minor_locator(mdates.YearLocator(base=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Histogram of mean monthly rainfall
    rainMonths = np.array([x.month for x in rainDates])
    rainMeans = np.zeros(12, dtype=np.float32)
    for m in range(1, 13):
        rainMeans[m-1] = np.mean(rainData["Mean_rain"][(rainMonths == m) &
                            (rainDates >= datetime.date(2003, month=1, day=1))])
    ax4 = plt.axes([0.63, 0.83, 0.22, 0.14])
    bars = ax4.bar(range(12), rainMeans, color='lightskyblue', width=1, align='center')
    ax4.tick_params(axis='y', length=2, pad=2, labelsize=8)
    ax4.tick_params(axis='x', length=1, pad=1)
    ax4.set_ylabel('Mean monthly\nprecipitation (mm)', fontsize=8, rotation=0, labelpad=1,
                   horizontalalignment='right', verticalalignment='center')
    ax4.set_xticks(range(12))
    ax4.set_xticklabels([])
    ax4.set_xlim([-0.5, 11.5])
    if np.max(rainMeans) < 50.0:
        ax4.set_ylim((0.0, 50.0))
    plt.setp(ax4.spines.values(), lw=0.5)
    
    # Histogram of fire frequency by month
    burnMonths = np.array([x.month for x in burnDates])
    burnMeans = np.zeros(12, dtype=np.float32)
    for m in range(1, 13):
        burnMeans[m-1] = np.mean(burnData["Burnt_area_percent"][burnMonths == m])
    ax5 = ax4.twinx()
    bars = ax5.bar(range(12), burnMeans, color='red', alpha=0.5, width=1, align='center')
    ax5.yaxis.set_label_position("right")
    ax5.tick_params(axis='y', length=2, pad=2, labelsize=8)
    ax5.set_ylabel('Mean monthly\nburnt area (%)', fontsize=8, rotation=0, labelpad=1,
                   horizontalalignment='left', verticalalignment='center')
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=8)
    ax5.set_xlim([-0.5, 11.5])
    if np.max(burnMeans) < 1.0:
        ax5.set_ylim((0.0, 1.0))
    
    outDir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\timeseries_monthly"
    plt.savefig(os.path.join(outDir, r'%s.png'%duneArea), dpi=300)
    plt.close(fig)
    
    # Calculate fire season
    summer = burnMeans[11] + burnMeans[0] + burnMeans[1]
    autumn = burnMeans[2] + burnMeans[3] + burnMeans[4]
    winter = burnMeans[5] + burnMeans[6] + burnMeans[7]
    spring = burnMeans[8] + burnMeans[9] + burnMeans[10]
    if summer + autumn + winter + spring == 0:
        fireSeasonList.append('No burn')
    elif summer > max([autumn, winter, spring]):
        fireSeasonList.append('Summer')
    elif autumn > max([summer, winter, spring]):
        fireSeasonList.append('Autumn')
    elif winter > max([summer, autumn, spring]):
        fireSeasonList.append('Winter')
    elif spring > max([summer, autumn, winter]):
        fireSeasonList.append('Spring')
    else:
        fireSeasonList.append('No season')
    
    # Calculate PV season
    pvMonths = np.array([x.month for x in fcDates])
    pvMeans = np.zeros(12, dtype=np.float32)
    for m in range(1, 13):
        pvMeans[m-1] = np.mean(fcData["PV_mean"][pvMonths == m])
    summer = pvMeans[11] + pvMeans[0] + pvMeans[1]
    autumn = pvMeans[2] + pvMeans[3] + pvMeans[4]
    winter = pvMeans[5] + pvMeans[6] + pvMeans[7]
    spring = pvMeans[8] + pvMeans[9] + pvMeans[10]
    if summer > max([autumn, winter, spring]):
        pvSeasonList.append('Summer')
    elif autumn > max([summer, winter, spring]):
        pvSeasonList.append('Autumn')
    elif winter > max([summer, autumn, spring]):
        pvSeasonList.append('Winter')
    elif spring > max([summer, autumn, winter]):
        pvSeasonList.append('Spring')
    else:
        pvSeasonList.append('No season')
    
    # Calculate NPV season
    npvMonths = np.array([x.month for x in fcDates])
    npvMeans = np.zeros(12, dtype=np.float32)
    for m in range(1, 13):
        npvMeans[m-1] = np.mean(fcData["NPV_mean"][npvMonths == m])
    summer = npvMeans[11] + npvMeans[0] + npvMeans[1]
    autumn = npvMeans[2] + npvMeans[3] + npvMeans[4]
    winter = npvMeans[5] + npvMeans[6] + npvMeans[7]
    spring = npvMeans[8] + npvMeans[9] + npvMeans[10]
    if summer > max([autumn, winter, spring]):
        npvSeasonList.append('Summer')
    elif autumn > max([summer, winter, spring]):
        npvSeasonList.append('Autumn')
    elif winter > max([summer, autumn, spring]):
        npvSeasonList.append('Winter')
    elif spring > max([summer, autumn, winter]):
        npvSeasonList.append('Spring')
    else:
        npvSeasonList.append('No season')
    
    # Calculate rainfall season
    summer = rainMeans[11] + rainMeans[0] + rainMeans[1]
    autumn = rainMeans[2] + rainMeans[3] + rainMeans[4]
    winter = rainMeans[5] + rainMeans[6] + rainMeans[7]
    spring = rainMeans[8] + rainMeans[9] + rainMeans[10]
    if summer > max([autumn, winter, spring]):
        rainSeasonList.append('Summer')
    elif autumn > max([summer, winter, spring]):
        rainSeasonList.append('Autumn')
    elif winter > max([summer, autumn, spring]):
        rainSeasonList.append('Winter')
    elif spring > max([summer, autumn, winter]):
        rainSeasonList.append('Spring')
    else:
        rainSeasonList.append('No season')

# Write seasonality data to CSV
nameList = np.array(nameList)
areaList = np.array(areaList)
climateList = np.array(climateList)
rainSeasonList = np.array(rainSeasonList)
fireSeasonList = np.array(fireSeasonList)
pvSeasonList = np.array(pvSeasonList)
npvSeasonList = np.array(npvSeasonList)
with open('season_summary.csv', 'w') as f:
    f.write('name,area,climate,rain_season,fire_season,pv_season,npv_season\n')
    for i in range(areaList.size):
        f.write('%s,%s,%s,%s,%s,%s,%s\n'%(nameList[i],
                                          areaList[i], climateList[i],
                                          rainSeasonList[i], fireSeasonList[i],
                                          pvSeasonList[i], npvSeasonList[i]))
