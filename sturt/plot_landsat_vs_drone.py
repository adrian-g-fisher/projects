#!/usr/bin/env python
"""
"""
import os
import sys
import glob
import numpy as np
from osgeo import gdal, ogr
from rios import applier
from rios import rat
from scipy import ndimage
from datetime import datetime
import matplotlib.pyplot as plt

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out'}
plt.rcParams.update(params)


def read_csv(inCSV):
    """
    All csv files have droneAlive, droneDead, satPV, satNPV
    """
    data = []
    with open(inCSV, 'r') as f:
        f.readline()
        for line in f:
            data.append(line.strip().split(',')[3:])
    data = np.array(data).astype(np.float32)
    return(data)


workDir = r"C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change"
fc_v2 = read_csv(os.path.join(workDir, r"comparison_fc.csv"))
fc_v3 = read_csv(os.path.join(workDir, r"comparison_fc_v3.csv"))
fc_AZN = read_csv(os.path.join(workDir, r"comparison_fc_AZN.csv"))

fig = plt.figure(1)
fig.set_size_inches((8, 8))

rects  = [[[0.30, 0.70, 0.25, 0.25], [0.60, 0.70, 0.25, 0.25]],
          [[0.30, 0.40, 0.25, 0.25], [0.60, 0.40, 0.25, 0.25]],
          [[0.30, 0.10, 0.25, 0.25], [0.60, 0.10, 0.25, 0.25]]]

data = [fc_v2, fc_v3, fc_AZN]

for x in range(2):
    for y in range(3):
        ax = plt.axes(rects[y][x])
        drone = data[y][:, x]
        satellite = data[y][:, x+2]
        ax.set_xlim((0, 100))
        ax.set_ylim((0, 100))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        if y != 2:
            ax.set_xticklabels([])
        if x != 0:
            ax.set_yticklabels([])

        if x == 0 and y == 0:
            ax.set_ylabel('Version 2\nLandsat fractional\ncover (%)', fontsize=12)
        if x == 0 and y == 1:
            ax.set_ylabel('Version 3\nLandsat fractional\ncover (%)', fontsize=12)            
        if x == 0 and y == 2:
            ax.set_ylabel('AZN model\nLandsat fractional\ncover (%)', fontsize=12)
            
        if x == 0 and y == 2:
            ax.set_xlabel('                                              Drone fractional cover (%)', fontsize=12)
        
        ax.plot(drone, satellite, ls='', marker='.', markeredgecolor='0.5', markerfacecolor='None')
        ax.plot([0, 100], [0, 100], ls='-', color='k', lw=1)
        
        
fig.text(0.42, 0.97, 'Green vegetation', fontsize=12, horizontalalignment="center")
fig.text(0.72, 0.97, 'Dead vegetation', fontsize=12, horizontalalignment="center")

plt.savefig(os.path.join(workDir, "drone_satellite_comparison"), dpi=300)
plt.clf()