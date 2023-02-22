#!/usr/bin/env python
"""

conda activate geo

"""


import os
import sys
import numpy as np
import ternary
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (4, 3.7)

# Read in data 
fc_data = []
colours = []
with open(r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\landsat_models.csv', 'r') as f:
    f.readline()
    for line in f:
        living = float(line.strip().split(',')[3])
        dead = float(line.strip().split(',')[4])
        bare = float(line.strip().split(',')[5])
        fc_data.append([living, dead, bare])
        blue = float(line.strip().split(',')[6]) / 10000.0
        green = float(line.strip().split(',')[7]) / 10000.0
        red = float(line.strip().split(',')[8]) / 10000.0
        colours.append([red, green, blue])
fc_data = np.array(fc_data)
colours = np.array(colours)

# Scale colours from min-max to 0-1
minColours = np.min(colours, axis=0)
maxColours = np.max(colours, axis=0)
colours = (colours - minColours) / (maxColours - minColours)

# Make ternary plot of field measurements
fontsize = 12
offset = 0.15
fig, tax = ternary.figure(scale=100)
tax.boundary(linewidth=1)
tax.gridlines(color="grey", multiple=20, linewidth=0.5)
tax.left_axis_label("Bare soil (%)", offset=offset)
tax.right_axis_label("Dead vegetation (%)", offset=offset)
tax.bottom_axis_label("Living vegetation (%)", offset=offset)
tax._redraw_labels()
tax.ticks(axis='lbr', linewidth=1, multiple=20, offset=0.02)
tax.set_background_color(color="white")
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.scatter(fc_data, marker='o', s=10, c=colours)
plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\ternary.png')
plt.clf()