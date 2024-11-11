#!/usr/bin/env python

"""

Works in the water conda environment:
conda create -n water scipy matplotlib scikit-learn
 
"""

import os
import sys
import numpy as np
import string
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from scipy import stats, odr
from sklearn import metrics

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


def load_data(datafile):
    """
    This reads the csv file, seperates pure pixels, and returns a tuple of
    flattened arrays. The data has the format:
    00 ID site
    01 Latitude
    02 Longitude
    03 Water percentage (%)
    04 Red
    05 Blue
    06 Green
    07 Red Edge 1
    08 Red Edge 2
    09 NIR
    10 SWIR 2
    11 SWIR 3
    12 MNDWI
    13 MuWIC
    14 TWI
    15 Date polygon
    16 Date S2 image
    17 BaseMap
    18 Zone
    """
    ref = []
    plot = []
    zone = []
    reflect = []
    indexes = []
    with open(datafile, 'r') as f:
        f.readline()
        for line in f:
            l = line.split(',')
            ref.append(float(l[3]))
            plot.append(int(l[0]))
            zone.append(l[18].strip())
            reflect.append([float(l[4]), float(l[5]), float(l[6]),
                            float(l[7]), float(l[8]), float(l[9]),
                            float(l[10]), float(l[11])])  
            indexes.append([float(l[12]), float(l[13]), float(l[14])])
    ref = np.asarray(ref).astype(np.float32)
    reflect = np.asarray(reflect).astype(np.float32) / 10000.0
    reflect[reflect < 0] = 0
    plot = np.asarray(plot)
    zone = np.asarray(zone)
    indexes = np.asarray(indexes)
    
    # Make the plot number unique
    plot[zone == 'Witchelina'] += 45
    
    return (ref, reflect, plot, zone, indexes)


def calculate_models(reflect, indexes):
    """
    This calculates the index values for all models.
    """
    model_values = np.zeros((np.shape(indexes)[0], 7), dtype=np.float32)
    names = ['MNDWI', 'MuWIC', 'TWI', "WI_Fisher"]
    model_values[:, 0] = indexes[:, 0]
    model_values[:, 1] = indexes[:, 1]
    model_values[:, 2] = indexes[:, 2]
    # reflect[:, 0] is red    
    # reflect[:, 2] is green
    # reflect[:, 5] is nir
    # reflect[:, 6] is swir1
    # reflect[:, 7] is swir2
    c = [1.7204, 171, 3, -70, -45, -71]
    model_values[:, 3] = (c[0] + c[1]*reflect[:, 2] + c[2]*reflect[:,0] +
                          c[3]*reflect[:,5] + c[4]*reflect[:,6] +
                          c[5]*reflect[:,7])
    
    return (names, model_values)


def make_histograms(ref, index_data, output):

    fancy_names = [r'MNDWI', r'MuWI', r'TWI', r'WI']

    rectangles = [[0.06, 0.83, 0.89, 0.15],
                  [0.06, 0.58, 0.89, 0.15],
                  [0.06, 0.33, 0.89, 0.15],
                  [0.06, 0.08, 0.89, 0.15]]

    ranges = [[-0.75, 1], [-10, 25], [-4, 4], [-60, 40]]

    fig = plt.figure(1)
    fig.set_size_inches((6, 6))

    for i in range(4):
        index_i = index_data[:, i]
        water = index_i[ref >= 50]
        nonwater = index_i[ref < 50]
        num_bins = 100
        (water_hist, bins)    = np.histogram(water, bins=num_bins, range=ranges[i], density=True)
        (nonwater_hist, bins) = np.histogram(nonwater, bins=num_bins, range=ranges[i], density=True)
        bin_centres = (bins + (bins[1]-bins[0])/2)[:num_bins]
        ax = plt.axes(rectangles[i])
        ax.plot(bin_centres, water_hist, ls="-", lw=1, color="deepskyblue")
        ax.fill_between(bin_centres, water_hist, lw=0, color="deepskyblue", alpha=0.2)
        ax.plot(bin_centres, nonwater_hist, ls="-", lw=1, color="goldenrod")
        ax.fill_between(bin_centres, nonwater_hist, lw=0, color="goldenrod", alpha=0.2)
        ax.set_xlim(ranges[i])
        ax.set_ylim((0, max(np.max(water_hist), np.max(nonwater_hist))))
        ax.set_yticks([])
        ax.set_xlabel(fancy_names[i], fontsize=12, labelpad=1)
        ax.set_ylabel('Density', fontsize=12)
    
    plt.savefig(output)
    plt.clf()

# Where are the water pixels with values < -10 on Fowlers?

# Raed in data
ref = []
zone = []
reflect = []
indexes = []
easting_northing = []
ref_date = []
s2_date = []
with open('Percentage_Water_All_Dates.csv', 'r') as f:
    f.readline()
    for line in f:
        l = line.split(',')
        easting_northing.append([float(l[1]), float(l[2])])
        ref.append(float(l[3]))
        zone.append(l[18].strip())
        reflect.append([float(l[4]), float(l[5]), float(l[6]),
                        float(l[7]), float(l[8]), float(l[9]),
                        float(l[10]), float(l[11])])  
        indexes.append([float(l[12]), float(l[13]), float(l[14])])
        ref_date.append(l[15])
        s2_date.append(l[16])
ref = np.asarray(ref).astype(np.float32)
reflect = np.asarray(reflect).astype(np.float32) / 10000.0
reflect[reflect < 0] = 0
zone = np.asarray(zone)
indexes = np.asarray(indexes)
easting_northing = np.asarray(easting_northing)
ref_date = np.asarray(ref_date)
s2_date = np.asarray(s2_date)

# Calculate WI
names, indexes = calculate_models(reflect, indexes)

# Select only water pixels on Fowlers with WI values < -10
e_n = easting_northing[(ref >= 50) & (zone == 'Fowlers_Gap') & (indexes[:, 3] < -10), :]
ref_dates = ref_date[(ref >= 50) & (zone == 'Fowlers_Gap') & (indexes[:, 3] < -10)]
s2_dates = s2_date[(ref >= 50) & (zone == 'Fowlers_Gap') & (indexes[:, 3] < -10)]

# Write out a CSV
with open('problem_water_pixels.csv', 'w') as f:
    f.write('easting,northing,ref_date,s2_date\n')
    for i in range(ref_dates.size):
        line = '%i,%i,%s,%s\n'%(e_n[i, 1], e_n[i, 0], ref_dates[i], s2_dates[i])
        f.write(line)
        
sys.exit()

ref, reflect, plot, zone, indexes = load_data('Percentage_Water_All_Dates.csv')
names, model_values = calculate_models(reflect, indexes)
make_histograms(ref, model_values, "waterindex_histograms.png")
make_histograms(ref[zone == 'Fowlers_Gap'], model_values[zone == 'Fowlers_Gap', :], "waterindex_histograms_fowlers.png")
make_histograms(ref[zone == 'Witchelina'], model_values[zone == 'Witchelina', :], "waterindex_histograms_witchelina.png")