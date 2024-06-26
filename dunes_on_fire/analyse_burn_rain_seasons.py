#!/usr/bin/env python
"""
- Make a folder for each dune area
- For each dune area make graphs of monthly burnt area (%) vs previous seasonal rainfall (mm)
- Make graphs for 1-12 seasons prior to the burn, using different accumulation periods of 1-8 seasons



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

# Output
basedir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\gis_data\dune_areas_hesse\Analysis_2000-2022\burn_rain_plots'

# Iterate over dune areas
for j, duneArea in enumerate(duneAreas):
    
    # Create output dir
    outdir = os.path.join(basedir, duneArea)
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)
    
    # Read in rainfall data
    # Columns are ID, Date, Mean_rain, Stdev_rain, Pixel_count
    rainData = np.genfromtxt(rainCSVs[j], names=True, delimiter=',')
    rainDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                        month=int(str(x)[4:6]), day=15)
                                        for x in rainData["Date"]])

    # Read in burnt area data
    # Columns are ID, Date, Burnt_area_percent, Pixel_count
    burnData = np.genfromtxt(burntCSVs[j], names=True, delimiter=',')
    burnDates = np.array([datetime.date(year=int(str(x)[0:4]),
                                        month=int(str(x)[4:6]), day=15)
                                        for x in burnData["Date"]])
    
    # Calculate seasonal rainfall and burnt area
    season = []
    year = []
    rain = []
    burntarea = []
    for y in range(2000, 2024):
        for s in ['autumn', 'winter', 'spring', 'summer']:
            if s == 'autumn':
                yList = [y, y, y]
                mList = [3, 4, 5]
            elif s == 'winter':
                yList = [y, y, y]
                mList = [6, 7, 8]
            elif s == 'spring':
                yList = [y, y, y]
                mList = [9, 10, 11]
            elif s == 'summer':
                yList = [y, y+1, y+1]
                mList = [12, 1, 2]
                
            dList = []
            for i in range(3):
                d = datetime.date(year=yList[i], month=mList[i], day=15)
                if (d in rainDates) and (d in burnDates):
                    dList.append(d)
            if len(dList) == 3:
                season.append(s)
                year.append(y)
                rain.append(rainData["Mean_rain"][rainDates == dList[0]] +
                            rainData["Mean_rain"][rainDates == dList[1]] +
                            rainData["Mean_rain"][rainDates == dList[2]])
                burntarea.append(burnData["Burnt_area_percent"][burnDates == dList[0]] +
                                 burnData["Burnt_area_percent"][burnDates == dList[1]] +
                                 burnData["Burnt_area_percent"][burnDates == dList[2]])
    rain = np.array(rain).flatten()
    burntarea = np.array(burntarea).flatten()
    
    # Make plot of rainfall vs burnt area for different lags and accumulation periods
    for lag in range(13):
        
        for accum in range(1, 9):
        
            # Calculate total rainfall for the accumulation period
            if accum > 1:
                w = np.ones(accum)
                r = ndimage.convolve1d(rain, weights=w, mode='constant', origin=(len(w)-1)//-2)
            else:
                r = rain
            
            # Calculate r and b using lag and accum
            if lag == 0:
                r = r[accum-1:]
                b = burntarea[accum-1:]
            else:
                r = r[accum-1:][0: -lag]
                b = burntarea[accum-1:][lag:]

            fig = plt.figure(i)
            fig.set_size_inches((2.5, 2.5))
            area = " ".join(duneArea.split('_')[0:-1])
            if area == 'GSD':
                area = 'Great Sandy Desert'
            if area == 'GVD':
                area = 'Great Victoria Desert'
            climate = duneArea.split('_')[-1]
            fig.text(0.2, 0.86, "%s (%s)\nlag = %i seasons\naccumulation = %i seasons"%(area, climate, lag, accum))
            ax1 = plt.axes([0.25, 0.25, 0.6, 0.6])
            ax1.set_ylabel('Precipitation (mm)')
            ax1.set_xlabel('Burnt area (%)')
            ax1.plot(b, r, 'o', markersize=2, c='k')
            plt.savefig(os.path.join(outdir, r'%s_%02d_%02d.png'%(duneArea, lag, accum)), dpi=300)
            plt.close(fig)
            
            sys.exit()