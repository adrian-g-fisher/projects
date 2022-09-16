#!/usr/bin/env python
"""

This analyses a variety of fractional cover data for Fowlers Gap

"""


import os
import sys
import numpy as np
import ternary
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (4, 3.7)

# Read in star transect data 
idList = []
idYearList = []
fc_data = []
colours = []
with open("fg_star_transects.csv", 'r') as f:
    f.readline()
    for line in f:
        ID = line.strip().split(',')[0]
        Year = line.strip().split(',')[2][0:4]
        idYearList.append('%s_%s'%(ID, Year))
        PV = float(line.strip().split(',')[20])
        NPV = float(line.strip().split(',')[21])
        Bare = float(line.strip().split(',')[22])
        fc_data.append([PV, NPV, Bare])
        colours.append([Bare, PV, NPV]) # red, green, blue
fc_data = np.array(fc_data) * 100
idYearList = np.array(idYearList)
colours = np.array(colours)

# Read in landsat fc data
NAT_idYearList = []
NAT_fc_data = []
with open("seasonal_fc_extract.csv", 'r') as f:
    f.readline()
    for line in f:
        ID = line.strip().split(',')[0]
        Year = line.strip().split(',')[1].split('_')[2][1:5]
        NAT_idYearList.append('%s_%s'%(ID, Year))
        PV = float(line.strip().split(',')[5])
        NPV = float(line.strip().split(',')[7])
        Bare = float(line.strip().split(',')[3])
        NAT_fc_data.append([PV, NPV, Bare])
NAT_idYearList = np.array(NAT_idYearList)
NAT_fc_data = np.array(NAT_fc_data)

# Read in AZN landsat fc data
AZN_idYearList = []
AZN_fc_data = []
with open("seasonal_AZN_fc_extract.csv", 'r') as f:
    f.readline()
    for line in f:
        ID = line.strip().split(',')[0]
        Year = line.strip().split(',')[1].split('_')[2][1:5]
        AZN_idYearList.append('%s_%s'%(ID, Year))
        PV = float(line.strip().split(',')[5])
        NPV = float(line.strip().split(',')[7])
        Bare = float(line.strip().split(',')[3])
        AZN_fc_data.append([PV, NPV, Bare])
AZN_idYearList = np.array(AZN_idYearList)
AZN_fc_data = np.array(AZN_fc_data)

# Read in landsat fc v3 data
v3_idYearList = []
v3_fc_data = []
with open("seasonal_fc_v3_extract.csv", 'r') as f:
    f.readline()
    for line in f:
        ID = line.strip().split(',')[0]
        Year = line.strip().split(',')[1].split('_')[2][1:5]
        v3_idYearList.append('%s_%s'%(ID, Year))
        PV = float(line.strip().split(',')[5])
        NPV = float(line.strip().split(',')[7])
        Bare = float(line.strip().split(',')[3])
        v3_fc_data.append([PV, NPV, Bare])
v3_idYearList = np.array(v3_idYearList)
v3_fc_data = np.array(v3_fc_data)

# Check that field and satellite data are in the same order
for i, IDY in enumerate(idYearList):
    if (IDY != NAT_idYearList[i]) or (IDY != AZN_idYearList[i]) or (IDY != v3_idYearList[i]):
        print("Data does not line up")

# Remove some sites as they have no data in the AZN model
fc_data = fc_data[(idYearList != '3_2012') & (idYearList != '4_2012') &
                  (idYearList != '8_2013') & (idYearList != '9_2013'), :]
NAT_fc_data = NAT_fc_data[(idYearList != '3_2012') & (idYearList != '4_2012') &
                         (idYearList != '8_2013') & (idYearList != '9_2013'), :]
AZN_fc_data = AZN_fc_data[(idYearList != '3_2012') & (idYearList != '4_2012') &
                         (idYearList != '8_2013') & (idYearList != '9_2013'), :]
v3_fc_data = v3_fc_data[(idYearList != '3_2012') & (idYearList != '4_2012') &
                        (idYearList != '8_2013') & (idYearList != '9_2013'), :]
colours = colours[(idYearList != '3_2012') & (idYearList != '4_2012') &
                  (idYearList != '8_2013') & (idYearList != '9_2013'), :]

# Make ternary plot of field measurements
colours = np.array(colours)
fontsize = 12
offset = 0.15
fig, tax = ternary.figure(scale=100)
tax.boundary(linewidth=1)
tax.gridlines(color="grey", multiple=20, linewidth=0.5)
tax.left_axis_label("Bare soil and rock", offset=offset)
tax.right_axis_label("Non-photosynthetic vegetation", offset=offset)
tax.bottom_axis_label("Photosynthetic vegetation", offset=offset)
tax._redraw_labels()
tax.ticks(axis='lbr', linewidth=1, multiple=20, offset=0.02)
tax.set_background_color(color="white")
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.scatter(fc_data, marker='o', c=colours)
plt.savefig("ternary.png")
plt.clf()

# Make plot of field vs satellite data
# Make plots
fig = plt.figure(1)
fig.set_size_inches((6, 6))
rect = [[[0.17, 0.65, 0.2, 0.2],[0.40, 0.65, 0.2, 0.2],[0.63, 0.65, 0.2, 0.2]],
        [[0.17, 0.40, 0.2, 0.2],[0.40, 0.40, 0.2, 0.2],[0.63, 0.40, 0.2, 0.2]],
        [[0.17, 0.15, 0.2, 0.2],[0.40, 0.15, 0.2, 0.2],[0.63, 0.15, 0.2, 0.2]],]

ylabels = ["Fractional cover v2", "Fractional cover v3", "Arid model"]
data = [NAT_fc_data, v3_fc_data, AZN_fc_data]

for row in range(3):
    for col in range(3):
    
        xdata = fc_data[:, col] / 100.0
        ydata = data[row][:, col] / 100.0
        
        ax = plt.axes(rect[row][col])
        ax.set_ylim((-0.05, 1.05))
        ax.set_xlim((-0.05, 1.05))
        ax.plot(xdata, ydata, marker="o", lw=0, markeredgecolor="0.7",
                markerfacecolor="none", markersize=3)
        ax.plot([-0.05, 1.05], [-0.05, 1.05], ls='-', c='k', lw='1')
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        if row == 2:
            ax.set_xticklabels([0.0, 0.5, 1.0], fontsize=8)
            if col == 1:
                ax.set_xlabel("Field measured cover", fontsize=8)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(ylabels[row], fontsize=8)
            ax.set_yticklabels([0.0, 0.5, 1.0], fontsize=8)
        else:
            ax.set_yticklabels([])
        if row == 0:
            if col == 0:
                ax.set_title("Photosynthetic\nvegetation", fontsize=8)
            if col == 1:
                ax.set_title("Non-photosynthetic\nvegetation", fontsize=8)
            if col == 2:
                ax.set_title("Bare soil\nand rock", fontsize=8)
        error = ydata - xdata
        rmse = np.sqrt(np.mean((error)**2))
        n = ydata.size
        ax.text(0.02, 0.90, "RMSE = %.2f"%rmse, fontsize=8)
        ax.text(0.02, 0.78, "n = %i"%n, fontsize=8)
plt.savefig("validation.png")
plt.clf()