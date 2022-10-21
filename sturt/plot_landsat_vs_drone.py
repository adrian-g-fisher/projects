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

rects  = [[[0.20, 0.65, 0.25, 0.25], [0.60, 0.65, 0.25, 0.25]],
          [[0.20, 0.35, 0.25, 0.25], [0.60, 0.35, 0.25, 0.25]],
          [[0.20, 0.05, 0.25, 0.25], [0.60, 0.05, 0.25, 0.25]]]

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
        #if x != 
        
        ax.set_ylabel('Cover (%)', fontsize=12)
        ax.text(1988.5, 35, 'Green vegetation', fontsize=12, horizontalalignment="left")
        ax.plot(drone, satellite, ls='', marker='.', markeredgecolor='0.5', markerfacecolor='None')
    
plt.savefig(os.path.join(workDir, "drone_satellite_comparison"), dpi=300)
plt.clf()