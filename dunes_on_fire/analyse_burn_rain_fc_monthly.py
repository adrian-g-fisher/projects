#!/usr/bin/env python
"""
- Make a folder for each dune area
- For each dune area get monthly burnt area (%), monthly rainfall (mm), monthly
  PV (%) and NPV (%).
- Create a plot with cross-correlation of rainfall, PV, NPV, and burnt area
"""
import os
import sys
import glob
import datetime
import numpy as np
from scipy import ndimage
from scipy import stats
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

# Output
outdir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\burn_rain_plots'

# Create csv for summary output
outCsv = os.path.join(outdir, 'rainfall_cross_correlation.csv')
with open(outCsv, 'w') as f:
    f.write('area,climate,PV_lag,PV_correlation,NPV_lag,NPV_correlation,Burn_lag,Burn_correlation\n')

# Iterate over dune areas
for i, duneArea in enumerate(duneAreas):
    
    # Read in rainfall data
    # Columns are ID, Date, Mean_rain, Stdev_rain, Pixel_count
    rainData = np.genfromtxt(rainCSVs[i], names=True, delimiter=',')
    rainDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                        month=int(str(x)[4:6]), day=15)
                                        for x in rainData["Date"]])
    rainData = rainData["Mean_rain"]
    
    # Read in burnt area data
    # Columns are ID, Date, Burnt_area_percent, Pixel_count
    burnData = np.genfromtxt(burntCSVs[i], names=True, delimiter=',')
    burnDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                        month=int(str(x)[4:6]), day=15)
                                        for x in burnData["Date"]])
    burnData = burnData["Burnt_area_percent"]
    
    # Read in fractional cover data
    # Columns are Date, ID, Pixel_count, BS_mean, BS_std, PV_mean, PV_std, NPV_mean, NPV_std
    fcData = np.genfromtxt(fcCSVs[i], names=True, delimiter=',')
    fcDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                      month=int(str(x)[4:6]), day=15)
                                      for x in fcData["Date"]])
    
    # Check dates are all present for burn data
    minDate = datetime.date(year=2001, month=1, day=15)
    maxDate = datetime.date(year=2022, month=12, day=15)
    for y in range(2001, 2023):
        for m in range(1, 13):
            d = datetime.date(year=y, month=m, day=15)
            if (d >= minDate) & (d <= maxDate):
                if burnData[burnDates == d].size == 0:
                    burnData = np.append(burnData, 0)
                    burnDates = np.append(burnDates, d)
    indices = np.argsort(burnDates)
    burnData = burnData[indices]
    burnDates = burnDates[indices]
    
    # Check dates are all present for rain data
    minDate = datetime.date(year=1995, month=1, day=15)
    maxDate = datetime.date(year=2022, month=12, day=15)
    for y in range(1995, 2023):
        for m in range(1, 13):
            d = datetime.date(year=y, month=m, day=15)
            if (d >= minDate) & (d <= maxDate):
                if rainData[rainDates == d].size == 0:
                    rainData = np.append(rainData, 0)
                    rainDates = np.append(rainDates, d)
    indices = np.argsort(rainDates)
    rainData = rainData[indices]
    rainDates = rainDates[indices]
    
    # Get name and climate of dune field
    area = " ".join(duneArea.split('_')[0:-1])
    if area == 'GSD':
        area = 'Great Sandy Desert'
    if area == 'GVD':
        area = 'Great Victoria Desert'
    climate = duneArea.split('_')[-1]
    
    # Cross correlation
    nlags = 18 
    pv_rain = np.zeros((2, nlags+1), dtype=np.float32)
    npv_rain = np.zeros((2, nlags+1), dtype=np.float32)
    burnt_rain = np.zeros((2, nlags+1), dtype=np.float32)
    for lag in range(0, nlags+1):
        pv_rain[0, lag] = -1 * lag
        npv_rain[0, lag] = -1 * lag
        burnt_rain[0, lag] = -1 * lag
        minDate = datetime.date(year=2001, month=1, day=15)
        maxDate = datetime.date(year=2022, month=12, day=15)
        pv = fcData['PV_mean'][(fcDates >= minDate) & (fcDates <= maxDate)]
        npv = fcData['NPV_mean'][(fcDates >= minDate) & (fcDates <= maxDate)]
        burnt = burnData[(burnDates >= minDate) & (burnDates <= maxDate)]
        
        if lag > 0:
            lag_years = int(lag/12)
            lag_months = lag - (lag_years * 12)
            maxDate = datetime.date(year=2022-lag_years, month=12-lag_months, day=15)
            if lag_months == 0:
                minDate = datetime.date(year=2001-(lag_years), month=1, day=15)        
            else:
                minDate = datetime.date(year=2001-(lag_years+1), month=12-lag_months+1, day=15)
        rain = rainData[(rainDates >= minDate) & (rainDates <= maxDate)]
        
        pv_rain[1, lag] = stats.pearsonr(pv, rain).statistic
        npv_rain[1, lag] = stats.pearsonr(npv, rain).statistic
        if np.max(burnt) > 0:
            burnt_rain[1, lag] = stats.pearsonr(burnt, rain).statistic
    
    # Get the lag value of the max corrleation value
    line = '%s,%s'%(area, climate)
    for data in [pv_rain, npv_rain, burnt_rain]:
        c = np.max(data[1, :])
        l = data[0, :][data[1, :] == c][-1]
        line = '%s,%i,%.4f'%(line, l, c)
    with open(outCsv, 'a') as f:
        f.write('%s\n'%line)
    
    # Make plot of correlation vs lag
    fig = plt.figure(1)
    fig.set_size_inches((4, 1.5))
    axs = plt.axes([0.2, 0.3, 0.78, 0.55])
    axs.bar(pv_rain[0, :], pv_rain[1, :], color='darkgreen', width=1, alpha=0.5)
    axs.bar(npv_rain[0, :], npv_rain[1, :], color='saddlebrown', width=1, alpha=0.5)
    axs.bar(burnt_rain[0, :], burnt_rain[1, :], color='red', width=1, alpha=0.5)
    axs.axhline(c='k', lw=1)
    axs.set_ylabel("Pearson\ncorelation")
    axs.set_xlabel('Rainfall lag (months)')
    axs.set_xlim([-18.5, 0.5])
    axs.set_ylim([-1, 1])
    axs.set_xticks(range(-18, 3, 3))
    axs.set_yticks([-1, -0.5, 0, 0.5, 1])
    fig.text(0.2, 0.9, "%s (%s)"%(area, climate), ha="left")
    plt.savefig(os.path.join(outdir, r'%s.png'%duneArea), dpi=300)
    plt.close(fig)
