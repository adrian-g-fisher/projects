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
from scipy import stats


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)

# Read in data
csvfile = r'C:\Users\Adrian\Documents\witchelina\AWP_120_Final_csv.csv'
data = np.genfromtxt(csvfile, delimiter=',', names=True)
            
# Annual change
for cover in ['PV', 'NPV', 'Bare']:
    for stat in ['mean', 'min', 'max']:
        for date in range(1988, 2023):
            colname = '%s_%s_%s'%(cover, stat, date)
            fc = data[colname]
            d = data['Distance_to_Water'] / 1000.0
            colname = colname.replace('grazing', 'conservation')
            outplot = r'E:\witchelina_awp\shapefiles\plots\annual\%s\%s\%s.png'%(cover.lower(), stat, colname)
            fig = plt.figure()
            fig.set_size_inches((3, 3))
            ax = plt.axes([0.2, 0.2, 0.7, 0.7])
            ax.set_facecolor('k')
            ax.set_title(date, fontsize=10)
            h = ax.hist2d(d[~np.isnan(fc)], fc[~np.isnan(fc)], bins=[80, 80], range=[[0, 8], [0, 100]], cmap='Greys')
            ax.set_xlabel('Distance to water point (km)')            
            ax.set_ylabel('%s %s (%%)'%(stat.capitalize(), cover))
            ax.set_xlim([0, 8])
            ax.set_ylim([0, 100])
            #plt.colorbar(h[3], ax=ax)
            (slope, intercept, r, p, se) = stats.linregress(d[~np.isnan(fc)], fc[~np.isnan(fc)])
            ax.plot([0, 8], [intercept, 8*slope+intercept], ls='-', c='r', lw=0.5)
            plt.savefig(outplot, dpi=300)
            plt.close()

# Before-after comparison
for cover in ['PV', 'NPV', 'Bare']:
    for stat in ['mean', 'min', 'max']:
        for date in ['before_grazing', 'after_grazing']:
            colname = '%s_%s_%s'%(cover, stat, date)
            fc = data[colname]
            d = data['Distance_to_Water'] / 1000.0
            colname = colname.replace('grazing', 'conservation')
            outplot = r'E:\witchelina_awp\shapefiles\plots\before_after\%s\%s\%s.png'%(cover.lower(), stat, colname)
            fig = plt.figure()
            fig.set_size_inches((3, 3))
            ax = plt.axes([0.2, 0.2, 0.7, 0.7])
            ax.set_facecolor('k')
            ax.set_title(date.replace('_grazing', ' conservation').capitalize(), fontsize=10)
            h = ax.hist2d(d, fc, bins=[80, 80], range=[[0, 8], [0, 100]], cmap='Greys')
            ax.set_xlabel('Distance to water point (km)')            
            ax.set_ylabel('%s %s (%%)'%(stat.capitalize(), cover))
            ax.set_xlim([0, 8])
            ax.set_ylim([0, 100])
            #plt.colorbar(h[3], ax=ax)
            (slope, intercept, r, p, se) = stats.linregress(d, fc)
            ax.plot([0, 8], [intercept, 8*slope+intercept], ls='-', c='r', lw=0.5)
            plt.savefig(outplot, dpi=300)

# Now repeat for each paddock?