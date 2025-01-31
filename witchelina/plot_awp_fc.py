#!/usr/bin/env python
"""
This plots the fractional cover time series statistics against distance from
artifical water points.

y = data['NPV_mean_before_grazing']
dummy = pd.get_dummies(data['Paddock']).values
X = np.column_stack((data['Distance']/1000.0, dummy[:, 1:]))
X = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
# Slope =  1.0783 +/- 0.022 (standard error)

y = data['NPV_mean_after_grazing']
dummy = pd.get_dummies(data['Paddock']).values
X = np.column_stack((data['Distance']/1000.0, dummy[:, 1:]))
X = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
# Slope = 0.529 +/- 0.022 (standard error)

y = data['Bare_mean_before_grazing']
dummy = pd.get_dummies(data['Paddock']).values
X = np.column_stack((data['Distance']/1000.0, dummy[:, 1:]))
X = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
# Slope =  -0.8426 +/- 0.022 (standard error)

y = data['Bare_mean_after_grazing']
dummy = pd.get_dummies(data['Paddock']).values
X = np.column_stack((data['Distance']/1000.0, dummy[:, 1:]))
X = sm.add_constant(X, prepend=False)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
# Slope = -0.2637 +/- 0.022 (standard error)

"""


import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import statsmodels.api as sm


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)

# Read in data
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\redo_analyses_epsg3577\awp_analysis_epsg3577.csv'
data = np.genfromtxt(csvfile, delimiter=',', names=True, dtype=None, encoding=None)

x = data['Distance'] / 1000.0
y = data['NPV_mean_before_grazing']
result = stats.linregress(x, y)
print(result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr)

y = data['NPV_mean_after_grazing']
result = stats.linregress(x, y)
print(result.slope, result.intercept, result.rvalue, result.pvalue, result.stderr, result.intercept_stderr)

sys.exit()

# Before-after comparison for each paddock
outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\redo_analyses_epsg3577\plots\paddocks_before_after'

for paddock in np.unique(data['Paddock']):
    outdir_paddock = os.path.join(outbase, paddock.replace(' ', '_'))
    if os.path.exists(outdir_paddock) is False:
        os.mkdir(outdir_paddock)
    for cover in ['PV', 'NPV', 'Bare']:
        outdir_cover = os.path.join(outdir_paddock, cover)
        if os.path.exists(outdir_cover) is False:
            os.mkdir(outdir_cover)
        for stat in ['mean', 'min', 'max']:
            outdir_stat = os.path.join(outdir_cover, stat)
            if os.path.exists(outdir_stat) is False:
                os.mkdir(outdir_stat)
            for date in ['before_grazing', 'after_grazing']:
                colname = '%s_%s_%s'%(cover, stat, date)
                fc = data[colname][data['Paddock'] == paddock]
                d = data['Distance'][data['Paddock'] == paddock] / 1000.0
                colname = colname.replace('grazing', 'conservation')
                outplot = os.path.join(outdir_stat, r'%s_%s.png'%(paddock.replace(' ', '_'), colname))
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
                ax.text(4, 90, paddock, horizontalalignment='center')
                #plt.colorbar(h[3], ax=ax)
                (slope, intercept, r, p, se) = stats.linregress(d, fc)
                ax.plot([0, 8], [intercept, 8*slope+intercept], ls='-', c='r', lw=0.5)
                plt.savefig(outplot, dpi=300)
                plt.close()

# Before-after comparison
outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\redo_analyses_epsg3577\plots\before_after'
for cover in ['PV', 'NPV', 'Bare']:
    outdir_cover = os.path.join(outbase, cover)
    if os.path.exists(outdir_cover) is False:
        os.mkdir(outdir_cover)
    for stat in ['mean', 'min', 'max']:
        outdir_stat = os.path.join(outdir_cover, stat)
        if os.path.exists(outdir_stat) is False:
            os.mkdir(outdir_stat)
        for date in ['before_grazing', 'after_grazing']:
            colname = '%s_%s_%s'%(cover, stat, date)
            fc = data[colname]
            d = data['Distance'] / 1000.0
            colname = colname.replace('grazing', 'conservation')
            outplot = os.path.join(outdir_stat, r'%s.png'%colname)
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
            ax.text(1, 90, 'slope = %0.4f'%slope)
            plt.savefig(outplot, dpi=300)

sys.exit()

# Annual change
outbase = r'C:\Users\Adrian\OneDrive - UNSW\Documents\witchelina\awp_grazing_pressure\redo_analyses_epsg3577\plots\annual'
for cover in ['PV', 'NPV', 'Bare']:
    outdir_cover = os.path.join(outbase, cover)
    if os.path.exists(outdir_cover) is False:
        os.mkdir(outdir_cover)
    for stat in ['mean', 'min', 'max']:
        outdir_stat = os.path.join(outdir_cover, stat)
        if os.path.exists(outdir_stat) is False:
            os.mkdir(outdir_stat)
        for date in range(1988, 2023):
            colname = '%s_%s_%s'%(cover, stat, date)
            fc = data[colname]
            d = data['Distance'] / 1000.0
            outplot = os.path.join(outdir_stat, r'%s.png'%colname)
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
