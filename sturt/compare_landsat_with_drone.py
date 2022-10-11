#!/usr/bin/env python
"""
Drone imagery was always captured March-April (Autumn).

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


# Read in the shapefile attributes to match Id to IDENT
polyfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\WildDeserts_monitoringsitehectareplots.shp'
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(polyfile, 0)
layer = dataSource.GetLayer()
Id2Ident = {}
for feature in layer:
    Id = int(feature.GetField("Id"))
    Ident = feature.GetField("Ident")
    Id2Ident[Id] = Ident
layer.ResetReading()


# Read in the drone veg cover data
droneData = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\FinalDeadAlive2018to2022.csv'
droneIdent = []
droneYear = []
droneType = []
dronePix = []
with open(droneData, 'r') as f:
    f.readline()
    for line in f:
        droneIdent.append(line.split(',')[5])
        droneYear.append(int(line.split(',')[1]))
        droneType.append(line.split(',')[7])
        dronePix.append(float(line.split(',')[6]))
droneIdent = np.array(droneIdent)
droneYear = np.array(droneYear)
droneType = np.array(droneType)
dronePix = np.array(dronePix)


# Re-arrange data and calculate living and dead proportions without shadows
dIdent = []
dYear = []
dAlive = []
dDead = []
years = np.unique(droneYear)
idents = np.unique(droneIdent)
fudgefactor = 1.00 # Proportion of PV that should be NPV
for y in years:
    for i in idents:
        if np.size(dronePix[(droneType == 'Alive') & (droneYear == y) & (droneIdent == i)]) > 0:
            alive = dronePix[(droneType == 'Alive') & (droneYear == y) & (droneIdent == i)][0]
            dead = dronePix[(droneType == 'Dead') & (droneYear == y) & (droneIdent == i)][0]
            background = dronePix[(droneType == 'Background') & (droneYear == y) & (droneIdent == i)][0]
            total = alive + dead + background
            alivePercent = 100 * alive / float(total)
            deadPercent = 100 * dead / float(total)
            alive_component = alivePercent * fudgefactor
            dead_component = (alivePercent * (1 - fudgefactor)) + deadPercent
            dIdent.append(i)
            dYear.append(y)        
            dAlive.append(alive_component)
            dDead.append(dead_component)
dIdent = np.array(dIdent)
dYear = np.array(dYear)
dAlive = np.array(dAlive)
dDead = np.array(dDead)


def read_csvfile(csvFile):
    """
    Reads the data from the csvfile and returns a numpy array.
    """
    Date = []
    i = []
    pixels = []
    mean_bare = []
    std_bare = []
    mean_green = []
    std_green = []
    mean_dead = []
    std_dead = []
    with open(csvFile, "r") as f:
        f.readline()
        for line in f:
            line = line.split(",")
            
            d = line[0]
            year = float(d[:4])
            month = float(d[4:6])
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            d = year + ((month+0.5)/12.0)
            Date.append(d)
            
            i.append(float(line[1]))
            pixels.append(float(line[2]))
            mean_bare.append(float(line[3]))
            std_bare.append(float(line[4]))
            mean_green.append(float(line[5]))
            std_green.append(float(line[6]))
            mean_dead.append(float(line[7]))
            std_dead.append(float(line[8]))
            
    data = np.array([i,Date,pixels,mean_bare,std_bare,mean_green,std_green,mean_dead,std_dead])
    data = data[:, data[1, :].argsort()]
    
    return data


def make_plot(csvfile, sumFile):
    """
    This makes the plot using the matplotlib graphing library.
    """
    fcdata = read_csvfile(csvfile)
    Id = fcdata[0, :][0]
    Ident = Id2Ident[Id]
    
    droneDates = dYear[dIdent == Ident] + (4.5/12.0)
    droneAlive = dAlive[dIdent == Ident] 
    droneDead = dDead[dIdent == Ident]
    
    dates = fcdata[1, :]
    fig = plt.figure(1)
    fig.set_size_inches((8, 6))
    rects  = [[0.1, 0.55, 0.87, 0.35],
              [0.1, 0.10, 0.87, 0.35]]
    fig.text(0.5, 0.97, 'Site ID %i (%s)'%(Id, Ident), fontsize=12, horizontalalignment="center")
    axGreen = plt.axes(rects[0])    
    axDead = plt.axes(rects[1])
    
    axGreen.set_xlim((1987, 2023))
    axGreen.set_ylim((0, 40))
    axGreen.set_xticklabels([])
    axGreen.set_yticks([0, 10, 20, 30, 40])
    axGreen.set_ylabel('Cover (%)', fontsize=12)
    axGreen.text(1988.5, 35, 'Green vegetation', fontsize=12, horizontalalignment="left")
    axGreen.fill_between(dates, fcdata[5, :]-fcdata[6, :], fcdata[5, :]+fcdata[6, :],
                         alpha=0.3, facecolor='k', linewidth=0.0, edgecolor='k')
    axGreen.plot(dates, fcdata[5, :], color='k', linewidth=1)
    
    axDead.set_xlim((1987, 2023))
    axDead.set_ylim((0, 80))
    axDead.set_xlabel('Years', fontsize=12)
    axDead.set_yticks([0, 20, 40, 60, 80])
    axDead.set_ylabel('Cover (%)', fontsize=12)
    axDead.text(1988.5, 70, 'Dead vegetation', fontsize=12, horizontalalignment="left")
    axDead.fill_between(dates, fcdata[7, :]-fcdata[8, :], fcdata[7, :]+fcdata[8, :],
                        alpha=0.3, facecolor='k', linewidth=0.0, edgecolor='k')
    axDead.plot(dates, fcdata[7, :], color='k', linewidth=1)
    
    axGreen.plot(droneDates, droneAlive, ls='', marker='o', markeredgecolor='r', markerfacecolor='r')
    axDead.plot(droneDates, droneDead, ls='', marker='o', markeredgecolor='r', markerfacecolor='r')
    
    plt.savefig(csvfile.replace('.csv', '_%s.png'%Ident.replace(' ', '_')), dpi=300)
    plt.clf()
    
    # Add to summary file
    with open(sumFile, 'a') as f:
        for i, iDate in enumerate(droneDates):
            iAlive = droneAlive[i]
            iDead = droneDead[i]
            f.write('%i,%s,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(Id, Ident, iDate, iAlive, iDead,
                                                        fcdata[5, :][dates == iDate],
                                                        fcdata[7, :][dates == iDate]))

# FC Version 2
sumFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\comparison_fc.csv'
csvDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\timeseries_fc'

# FC Version 3
#sumFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\comparison_fc_v3.csv'
#csvDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\timeseries_fc_v3'

# FC Arid Zone
#sumFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\comparison_fc_AZN.csv'
#csvDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\timeseries_fc_AZN'

# Create summary file
with open(sumFile, 'w') as f:
    f.write('Id,Ident,date,droneAlive,droneDead,satPV,satNPV\n')

# Make the graphs and summarise data
for csvfile in glob.glob(os.path.join(csvDir, '*_timeseries.csv')):
    make_plot(csvfile, sumFile)

print("Completed plots")