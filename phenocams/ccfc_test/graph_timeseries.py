#!/usr/bin/env python

import glob
import os, sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}

# Read in ccfc data
csvFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\ccfc_timeseries.csv'
ccfc_dates = []
ccfc_gcc = []
ccfc_ndvi = []
with open(csvFile, 'r') as f:
    f.readline()
    for line in f:
        (d, rcc, gcc, bcc, ndvi) = line.strip().split(',')
        ccfc_dates.append(datetime.date(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:])))
        ccfc_gcc.append(float(gcc))
        ccfc_ndvi.append(float(ndvi))
ccfc_dates = np.array(ccfc_dates)
ccfc_gcc = np.array(ccfc_gcc)
ccfc_ndvi = np.array(ccfc_ndvi)

# Read in Swift data
csvFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\phenocams\silcrete_lodge_test\swift_timeseries.csv'
swift_dates = []
swift_gcc = []
with open(csvFile, 'r') as f:
    f.readline()
    for line in f:
        (d, rcc, gcc, bcc) = line.strip().split(',')
        swift_dates.append(datetime.date(year=int(d[:4]), month=int(d[4:6]), day=int(d[6:])))
        swift_gcc.append(float(gcc))
swift_dates = np.array(swift_dates)
swift_gcc = np.array(swift_gcc)
minDate = datetime.date(year=2022, month=6, day=16)
maxDate = datetime.date(year=2022, month=8, day=2)
swift_gcc = swift_gcc[(swift_dates >= minDate) & (swift_dates <= maxDate)]
swift_dates = swift_dates[(swift_dates >= minDate) & (swift_dates <= maxDate)]

# Make plot
days = (maxDate - minDate).days
days = np.arange(0, days+1, 1)
xticks = list(range(0, 50, 5)) + [47]
xticklabels = [''] * len(xticks)
xticklabels[0] = minDate
xticklabels[-1] = maxDate

fig = plt.figure()
fig.set_size_inches((6, 2))
ax = plt.axes([0.15, 0.22, 0.75, 0.75])
ax.set_ylabel('GCC')
ax.set_xlabel('48 days')
ax.set_xlim([0, 47])
ax.set_ylim([0.35, 0.50])
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.grid(which='major', axis='x', c='0.9')
ax.plot(days, ccfc_gcc, color='0.5', linewidth=1)
plt.savefig(r'ccfc_gcc.png', dpi=300)

fig = plt.figure()
fig.set_size_inches((6, 2))
ax = plt.axes([0.15, 0.22, 0.75, 0.75])
ax.set_ylabel('NDVI')
ax.set_xlabel('48 days')
ax.set_xlim([0, 47])
ax.set_ylim([0.35, 0.55])
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.grid(which='major', axis='x', c='0.9')
ax.plot(days, ccfc_ndvi, color='0.5', linewidth=1)
plt.savefig(r'ccfc_ndvi.png', dpi=300)

fig = plt.figure()
fig.set_size_inches((6, 2))
ax = plt.axes([0.15, 0.22, 0.75, 0.75])
ax.set_ylabel('GCC')
ax.set_xlabel('48 days')
ax.set_xlim([0, 47])
ax.set_ylim([0.35, 0.45])
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.grid(which='major', axis='x', c='0.9')
ax.plot(days, swift_gcc, color='0.5', linewidth=1)
plt.savefig(r'swift_gcc.png', dpi=300)