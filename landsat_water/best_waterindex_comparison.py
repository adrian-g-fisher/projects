#!/usr/bin/env python

"""

 This does the following:
 - reads in the dbf and dbg values for the validation data
 - calculates the modelled values for each validation pixel
 - determines the optimum threshold to classify water and non-water using ROC curves
 - creates a plot of ROC curves
 - outputs a text file with accuracy statistics using optimum thresholds 
 
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


def load_data(datafile):
    """
    This reads the text file, seperates pure pixels, seperates classes, and
    returns a list of flattened arrays for water and nonwater for each index.
    
    The data has the format:
     ads40 landsat plot ref wi2006 red_water green_water blue_water red_nonwater
     green_nonwater blue_nonwater dbf_b1 dbf_b2 dbf_b3 dbf_b4 dbf_b5 dbf_b7
     dbg_b1 dbg_b2 dbg_b3 dbg_b4 dbg_b5 dbg_b7
    
    """
    ads40 = []
    ref = []
    plot = []
    wi2006 = []
    dbf = []
    dbg = []
    water_red = []
    water_gre = []
    water_blu = []
    nonwater_red = []
    nonwater_gre = []
    nonwater_blu = []
    with open(datafile, 'r') as f:
        for line in f:
            if line != [] and string.split(line, ' ')[0] != 'ads40':
                l = string.split(line, ' ')
                ads40.append(l[0])
                plot.append(int(l[2]))
                ref.append(int(l[3]))
                wi2006.append(int(l[4]))
                water_red.append(int(l[5]))
                water_gre.append(int(l[6]))
                water_blu.append(int(l[7]))
                nonwater_red.append(int(l[8]))
                nonwater_gre.append(int(l[9]))
                nonwater_blu.append(int(l[10]))
                dbf.append([float(l[11]), float(l[12]), float(l[13]),
                            float(l[14]), float(l[15]), float(l[16])])
                dbg.append([float(l[17]), float(l[18]), float(l[19]),
                            float(l[20]), float(l[21]), float(l[22])])   
    
    ads40 = np.asarray(ads40)
    plot = np.asarray(plot)
    ref = np.asarray(ref).astype(np.float32)
    wi2006 = np.asarray(wi2006).astype(np.float32)
    dbf = np.asarray(dbf).astype(np.float32) * 10000
    dbg = np.asarray(dbg).astype(np.float32) * 10000
    water_red = np.asarray(water_red).astype(np.float32)
    water_gre = np.asarray(water_gre).astype(np.float32)
    water_blu = np.asarray(water_blu).astype(np.float32)
    water_ads_rgb = [water_red, water_gre, water_blu]
    nonwater_red = np.asarray(nonwater_red).astype(np.float32)
    nonwater_gre = np.asarray(nonwater_gre).astype(np.float32)
    nonwater_blu = np.asarray(nonwater_blu).astype(np.float32)
    nonwater_ads_rgb = [nonwater_red, nonwater_gre, nonwater_blu]
    
    return (ads40, ref, wi2006, dbf, dbg, water_ads_rgb, nonwater_ads_rgb, plot)


def calculate_models(wi2006, dbf, dbg):
    """
    This calculates the index values for all models.
    """
    model_values = np.zeros((np.shape(wi2006)[0], 7), dtype=np.float)
    stage = ["db8", "dbg", "dbg", "dbg", "dbg", "dbg", "dbg"]
    names = ["wi2006", "wi2015", "awei_sh", "awei_nsh", "ndwi_m", "ndwi_x", "tcw_crist"]
    
    model_values[:,0] = wi2006
    
    for x in range(1, 7):

        if stage[x] == "dbf":
            X = dbf
        else:
            X = dbg
        
        if names[x] == "wi2015":
            c = [1.7204, 0.0171, 0.0003, -0.0070, -0.0045, -0.0071]
            model_values[:, x] = (c[0] + c[1]*X[:,1] + c[2]*X[:,2] +
                                  c[3]*X[:,3] + c[4]*X[:,4] + c[5]*X[:,5])
        
        elif names[x] == "ndwi_m":
            model_values[:,x] = (X[:,1] - X[:,3]) / (X[:,1] + X[:,3] + 1)
            
        elif names[x] == "ndwi_x":
            model_values[:,x] = (X[:,1] - X[:,4]) / (X[:,1] + X[:,4] + 1)
        
        elif names[x] == "tcw_crist":
            c = [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]
            X = X/10000.0
            model_values[:,x] = (c[0]*X[:,0] + c[1]*X[:,1] + c[2]*X[:,2] +
                                   c[3]*X[:,3] + c[4]*X[:,4] + c[5]*X[:,5])
         
        elif names[x] == "awei_nsh":
            X = X/10000.0
            model_values[:,x] = 4 * (X[:,1] - X[:,4]) - (0.25*X[:,3] + 2.75*X[:,4])
        
        elif names[x] == "awei_sh":
            X = X/10000.0
            model_values[:,x] = X[:,0] + 2.5*X[:,1] - 1.5*(X[:,3] + X[:,4]) - 0.25*X[:,5]
    
    return (names, model_values)


def getErrors(y, z):
    """
    Makes an error matrix from reference (y) and prediction (z) vectors.
    """
    a = np.sum(np.where(y == -1, 0, np.where(z == -1, 0, 1)))
    b = np.sum(np.where(y ==  1, 0, np.where(z == -1, 0, 1))) # Commission
    c = np.sum(np.where(y == -1, 0, np.where(z ==  1, 0, 1))) # Omission
    d = np.sum(np.where(y ==  1, 0, np.where(z ==  1, 0, 1)))
    if a + b + c + d > 0:
        acc = 100 * (a + d) / float(a + b + c + d)
    else:
        acc = 0
    if a + c > 0:
        TPR = 100 * a / float(a + c)
    else:
        TPR = 0
    if b + d > 0:
        FPR = 100 * b / float(b + d)
    else:
        FPR = 0
    if a + b > 0:
        users = 100*a/float(a+b)
    else:
        users = 0
    
    return (acc, TPR, FPR, users)


def ROC_optimum(ref_data, index_data, names):
    """
    This iteratively calculates the FPR and TPR for different thresholds, it
    creates ROC curves, caluclates the area under the curve (AUC) and optimum
    threshold.
    """
    D = []
    for i in range(np.shape(index_data)[1]):
        
        # Subset index from all index_data
        index_i = index_data[:, i]
        
        # Subset pure water values and pure non-water values
        water = index_i[ref_data == 200]
        nonwater = index_i[ref_data <= 100]
        
        # Create reference values
        ref = np.concatenate([np.where(water >= -999, 1, 0),
                              np.where(nonwater >= -999, -1, 0)])

        # Vary threshold between min and max, with 1000 increments
        increment = (np.max(index_i) - np.min(index_i)) / 1000.0
        thresholds = np.zeros((1000), dtype=float)
        thresholds[0] = np.min(index_i)
        for t in range(1, 1000):
            thresholds[t] = thresholds[t-1] + increment
        
        d = []
        for thresh in thresholds:
            if np.median(water) < np.median(nonwater):
                pred = np.concatenate([np.where(water <= thresh, 1, -1),
                                       np.where(nonwater <= thresh, 1, -1)])   
            else:
                pred = np.concatenate([np.where(water >= thresh, 1, -1),
                                       np.where(nonwater >= thresh, 1, -1)]) 
            (acc, TPR, FPR, users) = getErrors(ref, pred)
            d.append([thresh, acc, TPR, FPR])
            
        d = np.asarray(d)
        D.append(d)
    
    # Calculate optimum threshold as point closest to top left corner
    optimum_thresholds = []
    for i, d in enumerate(D):
        dist = 100 - d[:,2] + d[:,3]
        t = d[:,0][dist == np.min(dist)]
        if len(t) == 1:
            optimum_thresholds.append(t)
        else:
            optimum_thresholds.append(t[-1])
    
    # Create graph
    fig = plt.figure(1)
    fig.set_size_inches((6, 4))
    rect  = [0.12, 0.12, 0.6, 0.85]
    ax = plt.axes(rect)
    ax.set_xlim((-0.5, 6.5))
    ax.set_ylim((93.5, 100.5))
    ax.set_xlabel('False positive rate (%)', fontsize=12)
    ax.set_ylabel('True positive rate (%)', fontsize=12)
    
    # Plot curves and optimum points for top ten models
    colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "k"]
    fancy_names = [r'$WI_{2006}$', r'$WI_{2015}$', r'$AWEI_{shadow}$',
                   r'$AWEI_{no shadow}$', r'$NDWI_{McFeeters}$', r'$NDWI_{Xu}$',
                   r'$TCW_{Crist}$']
    for i in range(7):
        d = D[i]
        ax.plot(d[:,3], d[:,2], ls="-", lw=1.5, color=colors[i], label=fancy_names[i])
        ax.plot(d[:,3][d[:,0] == optimum_thresholds[i]],
                d[:,2][d[:,0] == optimum_thresholds[i]], marker="o", color='k')
    
    plt.legend(loc=(1, 0.2), fontsize=12, frameon=False, handletextpad=0.2)
    plt.savefig("best_waterindex_roc_curves.png")
    plt.savefig("figure_4.eps")
    plt.clf()
    
    # Calculate area under the curve (AUC) using the trapezoidal rule
    AUC = []
    for i in range(7):
        fpr_tpr = D[i][:,2:4]/100.0
        if i != 0:
           fpr_tpr = np.flipud(fpr_tpr) # Flip array
        fpr_tpr = np.vstack([np.array([[0.0, 0.0]]), fpr_tpr, np.array([[1.0, 1.0]])])
        row_diff = np.diff(fpr_tpr, axis=0)
        unique_rows = np.ones(len(fpr_tpr), dtype='bool')
        unique_rows[1:] = (row_diff != 0).any(axis=1) 
        fpr_tpr = fpr_tpr[unique_rows]
        auc = metrics.auc(fpr_tpr[:, 1], fpr_tpr[:, 0]) * 100
        AUC.append(auc)
        
    return (optimum_thresholds, AUC)


def ROC_optimum_OVERALL(ref_data, index_data, names):
    """
    This iteratively calculates the acc+prods+users for different thresholds,
    graphs the result and determined the optimum threshold.
    """
    D = []
    for i in range(np.shape(index_data)[1]):
        
        # Subset index from all index_data
        index_i = index_data[:, i]
        
        # Subset pure water values and pure non-water values
        water = index_i[ref_data == 200]
        nonwater = index_i[ref_data <= 100]
        
        # Create reference values
        ref = np.concatenate([np.where(water >= -999, 1, 0),
                              np.where(nonwater >= -999, -1, 0)])

        # Vary threshold between min and max, with 1000 increments
        increment = (np.max(index_i) - np.min(index_i)) / 1000.0
        thresholds = np.zeros((1000), dtype=float)
        thresholds[0] = np.min(index_i)
        for t in range(1, 1000):
            thresholds[t] = thresholds[t-1] + increment
        
        d = []
        for thresh in thresholds:
            if np.median(water) < np.median(nonwater):
                pred = np.concatenate([np.where(water <= thresh, 1, -1),
                                       np.where(nonwater <= thresh, 1, -1)])   
            else:
                pred = np.concatenate([np.where(water >= thresh, 1, -1),
                                       np.where(nonwater >= thresh, 1, -1)]) 
            (acc, prods, FPR, users) = getErrors(ref, pred)
            d.append([thresh, acc, prods, users])
            
        d = np.asarray(d)
        D.append(d)
    
    # Calculate optimum threshold as point with greatest acc+prods+users
    optimum_thresholds = []
    for i, d in enumerate(D):
        overall = d[:,1] + d[:,2] + d[:,3]
        t = d[:,0][overall == np.max(overall)]
        if len(t) == 1:
            optimum_thresholds.append(t)
        else:
            optimum_thresholds.append(t[-1])
    
    AUC = [0, 0, 0, 0, 0, 0, 0]
        
    return (optimum_thresholds, AUC)


def calculate_statistics(ref_data, index_data, thresholds):
    """
    This calculates error matrix statistics for the optimum thresholds.
    """
    
    stats = []
    
    for i in range(np.shape(index_data)[1]):
        
        # Subset index from all index_data
        index_i = index_data[:, i]
        
        # Subset pure water values and pure non-water values
        water = index_i[ref_data == 200]
        nonwater = index_i[ref_data <= 100]
        
        # Create reference values
        ref = np.concatenate([np.where(water >= -999, 1, 0),
                              np.where(nonwater >= -999, -1, 0)])
        
        if np.median(water) < np.median(nonwater):
              pred = np.concatenate([np.where(water <= thresholds[i], 1, -1),
                                     np.where(nonwater <= thresholds[i], 1, -1)])      
        else:
              pred = np.concatenate([np.where(water >= thresholds[i], 1, -1),
                                     np.where(nonwater >= thresholds[i], 1, -1)])
        
        (acc, prods, fpr, users) = getErrors(ref, pred)
        i_stats = [thresholds[i], acc, prods, users, fpr]
        stats.append(i_stats)
        
    return np.asarray(stats).astype(float)


def calculate_operational_statistics(ref_data, index_data, thresholds, ads40, a_red):
    """
    This calculates error matrix statistics for operational use: all pixels
    excluding ocean, cloud shadow and topo-shadow
    """
    
    # Get ocean pixels
    ocean = np.zeros_like(ref_data)
    ocean_ads = ['sydney_l5_ref.img', 'wingham_l5_ref.img', 'wingham_l7_ref.img']
    for a in ocean_ads:
        n = np.where(ads40 == a, np.where(ref_data == 200, np.where(a_red == 0, 1, 0), 0), 0)
        ocean[n == 1] = 1
    
    # Create subsets
    subset_water = np.where(ref_data >= 150, 1, 0)
    subset_water[ocean == 1] = 0
    subset_non = np.where(ref_data < 150, 1, 0)
    subset_non[ref_data == 3] = 0
    subset_non[ref_data == 4] = 0
        
    stats = []
    
    for i in range(np.shape(index_data)[1]):
        
        # Subset index from all index_data
        index_i = index_data[:, i]
        
        # Subset water values and non-water values
        # excluding ocean, cloud shadow and topo-shadow
        water = index_i[subset_water == 1]
        nonwater = index_i[subset_non == 1]
        
        # Create reference values
        ref = np.concatenate([np.where(water >= -999, 1, 0),
                              np.where(nonwater >= -999, -1, 0)])
        
        if i == 0:
              pred = np.concatenate([np.where(water <= thresholds[i], 1, -1),
                                     np.where(nonwater <= thresholds[i], 1, -1)])      
        else:
              pred = np.concatenate([np.where(water >= thresholds[i], 1, -1),
                                     np.where(nonwater >= thresholds[i], 1, -1)])
        
        (acc, prods, fpr, users) = getErrors(ref, pred)
        i_stats = [acc, prods, users, fpr]
        stats.append(i_stats)

    return np.asarray(stats).astype(float)


def make_histograms(ref, index_data, thresh):

    fancy_names = [r'$WI_{2006}$', r'$WI_{2015}$', r'$AWEI_{shadow}$',
                   r'$AWEI_{no shadow}$', r'$NDWI_{McFeeters}$', r'$NDWI_{Xu}$',
                   r'$TCW_{Crist}$']

    rectangles = [[0.06, 0.912, 0.89, 0.085], [0.06, 0.770, 0.89, 0.085],
                  [0.06, 0.628, 0.89, 0.085], [0.06, 0.486, 0.89, 0.085],
                  [0.06, 0.344, 0.89, 0.085], [0.06, 0.202, 0.89, 0.085],
                  [0.06, 0.060, 0.89, 0.085]]

    ranges = [[40, 100], [-50, 50], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-0.25, 0.1]]

    fig = plt.figure(1)
    fig.set_size_inches((5, 8))

    for i in range(7):
        index_i = index_data[:, i]
        water = index_i[ref == 200]
        nonwater = index_i[ref <= 100]
        num_bins = 60
        (water_hist, bins)    = np.histogram(water, bins=num_bins, range=ranges[i], density=False)
        (nonwater_hist, bins) = np.histogram(nonwater, bins=num_bins, range=ranges[i], density=False)
        bin_centres = (bins + (bins[1]-bins[0])/2)[:num_bins]
        ax = plt.axes(rectangles[i])
        ax.plot(bin_centres, water_hist, ls="-", lw=2, color="black")
        ax.plot(bin_centres, nonwater_hist, ls="-", lw=2, color=[0.7, 0.7, 0.7])
        ax.plot([thresh[i], thresh[i]], [0, max(np.max(water_hist), np.max(nonwater_hist))], ls="--", lw=2, color="red")
        ax.set_xlim(ranges[i])
        ax.set_ylim((0, max(np.max(water_hist), np.max(nonwater_hist))))
        ax.set_yticks([])
        ax.set_xlabel(fancy_names[i], fontsize=12, labelpad=1)
        ax.set_ylabel('Density', fontsize=12)

    plt.savefig("best_waterindex_histograms.png")
    plt.savefig("figure_5.eps")
    plt.clf()


def coloured_histograms(ref, dbg, index_data, thresh):

    fancy_names = [r'$WI_{2006}$', r'$WI_{2015}$', r'$AWEI_{shadow}$',
                   r'$AWEI_{no shadow}$', r'$NDWI_{McFeeters}$', r'$NDWI_{Xu}$',
                   r'$TCW_{Crist}$']

    rectangles = [[0.06, 0.912, 0.89, 0.085], [0.06, 0.770, 0.89, 0.085],
                  [0.06, 0.628, 0.89, 0.085], [0.06, 0.486, 0.89, 0.085],
                  [0.06, 0.344, 0.89, 0.085], [0.06, 0.202, 0.89, 0.085],
                  [0.06, 0.060, 0.89, 0.085]]

    ranges = [[40, 100], [-50, 50], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-0.25, 0.1]]

    fig = plt.figure(1)
    fig.set_size_inches((5, 8))

    for i in range(7):
        index_i = index_data[:, i]
        water = index_i[ref == 200]
        nonwater = index_i[ref <= 100]
        
        # Sort into clear and coloured water
        # clear water has b1 - b2 > -0.011 and
        #                 b1 + b2 <  0.053
        diff = dbg[:,0]/10000.0 - dbg[:,1]/10000.0
        add = dbg[:,0]/10000.0 + dbg[:,1]/10000.0
        watertype = np.where(diff > -0.011, np.where(add < 0.053, 0, 1), 1)
        watertype = watertype[ref == 200]
        clear = water[watertype == 0]
        coloured = water[watertype == 1]
        
        num_bins = 60
        (clear_hist, bins)    = np.histogram(clear, bins=num_bins, range=ranges[i], density=False)
        (coloured_hist, bins) = np.histogram(coloured, bins=num_bins, range=ranges[i], density=False)
        (nonwater_hist, bins) = np.histogram(nonwater, bins=num_bins, range=ranges[i], density=False)
        bin_centres = (bins + (bins[1]-bins[0])/2)[:num_bins]
        ax = plt.axes(rectangles[i])
        ax.plot(bin_centres, clear_hist, ls="-", lw=2, color="blue")
        ax.plot(bin_centres, coloured_hist, ls="-", lw=2, color="green")
        ax.plot(bin_centres, nonwater_hist, ls="-", lw=2, color=[0.7, 0.7, 0.7])
        ax.plot([thresh[i], thresh[i]], [0, max(np.max(clear_hist), np.max(coloured_hist), np.max(nonwater_hist))], ls="--", lw=2, color="red")
        ax.set_xlim(ranges[i])
        ax.set_ylim((0, max(np.max(clear_hist), np.max(coloured_hist), np.max(nonwater_hist))))
        ax.set_yticks([])
        ax.set_xlabel(fancy_names[i], fontsize=12, labelpad=1)
        ax.set_ylabel('Density', fontsize=12)

    plt.savefig("best_coloured_waterindex_histograms.png")
    plt.clf()


def calculate_errors(ads40, a_red, ref, dbg, model_values, thresholds):
    
    dbg = dbg/10000.0
   
    # Classify water
    water_type = np.zeros_like(ads40)
    r = np.where(ref < 200, np.where(ref > 100, 1, 0), 0)
    water_type[r == 1] = 'mixed'
    water_type[ref <= 100] = 'non-water'
    ocean_ads = ['sydney_l5_ref.img', 'wingham_l5_ref.img', 'wingham_l7_ref.img']
    for a in ocean_ads:
        n = np.where(ads40 == a, np.where(ref == 200, np.where(a_red == 0, 1, 0), 0), 0)
        water_type[n == 1] = 'ocean'
    clear = np.where(dbg[:,0] - dbg[:,1] > -0.011, np.where(dbg[:,0] + dbg[:,1] < 0.053, 1, 0), 0)
    water_type = np.where(clear == 1, np.where(water_type == '0', 'clear', water_type), water_type)

    clear_diff = (dbg[:,0] - dbg[:,1])[water_type == "clear"]
    clear_sum = (dbg[:,0] + dbg[:,1])[water_type == "clear"]
    total = dbg[:,1] + dbg[:,2]
    diff = dbg[:,1] - dbg[:,2]
    darkgreen = np.where((total - diff * 6) < 0, 1, 0)
    green = np.where((total - diff * 6) >= 0, np.where((total - diff * 28) < 0, 1, 0), 0)
    greenbrown = np.where((total - diff * 28) >= 0, np.where((total + diff * 28) >= 0, 1, 0), 0)
    brown = np.where((total + diff * 28) < 0, np.where((total + diff * 6) >= 0, 1, 0), 0)
    darkbrown = np.where((total + diff * 6) < 0, 1, 0)
    water_type = np.where(water_type == '0', np.where(darkgreen == 1, 'dark-green', water_type), water_type)
    water_type = np.where(water_type == '0', np.where(green == 1, 'green', water_type), water_type)
    water_type = np.where(water_type == '0', np.where(greenbrown == 1, 'green-brown', water_type), water_type)
    water_type = np.where(water_type == '0', np.where(brown == 1, 'brown', water_type), water_type)
    water_type = np.where(water_type == '0', np.where(darkbrown == 1, 'dark-brown', water_type), water_type)
    
    # Calculate omission errors
    watertypes = ["ocean", "clear", "dark-green", "green", "green-brown", "brown", "dark-brown"]
    omi = np.zeros((7, 8), dtype=float)
    for i in range(7):
        t = thresholds[i]
        index = model_values[:,i]
        for j, w in enumerate(watertypes):
            I = index[water_type == w]
            if i == 0:
                error = np.shape(I[I > t])[0]
            else:
                error = np.shape(I[I < t])[0]
            n = np.shape(water_type[water_type == w])[0]
            if n == 0:
                e = 0
            else:
                e = 100 * (error / float(n))
            omi[j, 0] = n
            omi[j, i+1] = e
    
    # Classify land
    land_type = np.zeros_like(water_type)
    land_type = np.where(water_type == 'non-water', np.where(ref == 100, 'land', land_type), land_type)
    land_type = np.where(water_type == 'non-water', np.where(ref == 3, 'cloud-shadow', land_type), land_type)
    land_type = np.where(water_type == 'non-water', np.where(ref == 4, 'topographic-shadow', land_type), land_type)
    land_type = np.where(water_type == 'non-water', np.where(ref == 5, 'quarries', land_type), land_type)
    land_type = np.where(water_type == 'non-water', np.where(ref == 6, 'urban', land_type), land_type)
    s = (dbg[:,3] + dbg[:,2])
    s = np.where(s == 0, np.min(s[s != 0]), s)
    ndvi = (dbg[:,3] - dbg[:,2]) / s.astype(float)
    land_type = np.where(water_type == 'non-water', np.where(ref == 100, np.where(ndvi < 0.2, "bare ground", land_type), land_type), land_type)
    land_type = np.where(water_type == 'non-water', np.where(ref == 100, np.where(ndvi >= 0.2, np.where(ndvi < 0.5, "sparse vegetation", land_type), land_type), land_type), land_type)
    land_type = np.where(water_type == 'non-water', np.where(ref == 100, np.where(ndvi >= 0.5, "dense vegetation", land_type), land_type), land_type)
    
    sample = ndvi[land_type == "dense vegetation"]
    with open("ndvi_dense_vegetation_analysis.txt", "w") as test:
        for sam in range(np.size(sample)):
            test.write("%12f\n"%sample[sam])
    
    # Calculate commission errors
    landtypes = ['bare ground', 'sparse vegetation', 'dense vegetation', 'cloud-shadow', 'topographic-shadow', 'quarries', 'urban']
    com = np.zeros((7, 8), dtype=float)
    for i in range(7):
        t = thresholds[i]
        index = model_values[:,i]
        for j, l in enumerate(landtypes):
            I = index[land_type == l]
            if i == 0:
                error = np.shape(I[I < t])[0]
            else:
                error = np.shape(I[I > t])[0]        
            n = np.shape(land_type[land_type == l])[0]
            e = 100 * (error / float(n))
            com[j, 0] = n
            com[j, i+1] = e
    
    return (omi, com)


def mixed_ROC_optimum(ref_data, index_data, thresholds):
    """
    This iteratively calculates the optimum threshold for classifying water and
    nonwater from ROC curves.
    """
    D = []
    m = np.where(ref_data > 100, np.where(ref_data < 200, 1, 0), 0)
    ref_data = ref_data[m == 1]
    
    for i in range(np.shape(index_data)[1]):
        
        d = []
        
        # Subset index from all index_data
        index_i = index_data[:, i]
        
        # Subset pure water values and pure non-water values
        mixed = index_i[m == 1]
        
        # Classify index using optimum threshold
        if i == 0:
            water = ref_data[mixed <= thresholds[i]] - 100
            nonwater = ref_data[mixed > thresholds[i]] - 100
            
        else:
            water = ref_data[mixed >= thresholds[i]] - 100
            nonwater = ref_data[mixed < thresholds[i]] - 100
        
        pred = np.concatenate([np.where(water >= -999, 1, 0),
                              np.where(nonwater >= -999, -1, 0)])

        for thresh in range(101):
            ref = np.concatenate([np.where(water >= thresh, 1, -1),
                                  np.where(nonwater >= thresh, 1, -1)])      
            (acc, TPR, FPR, users) = getErrors(ref, pred)
            d.append([thresh, acc, TPR, FPR, users])
        
        d = np.asarray(d)
        D.append(d)
    
    # Calculate optimum threshold as point closest to top left corner
    optimum_thresholds = []
    for i, d in enumerate(D):
        dist = 100 - d[:,2] + d[:,3]
        t = d[:,0][dist == np.min(dist)]
        if len(t) == 1:
            optimum_thresholds.append(t)
        else:
            print i, t
            optimum_thresholds.append(t[-1])
        
    # Create graph
    fig = plt.figure(1)
    fig.set_size_inches((6, 4))
    rect  = [0.12, 0.12, 0.6, 0.85]
    ax = plt.axes(rect)
    ax.set_xlim((4.5, 25.5))
    ax.set_ylim((54.5, 75.5))
    ax.set_xlabel('False positive rate (%)', fontsize=12)
    ax.set_ylabel('True positive rate (%)', fontsize=12)
    
    # Plot curves, optimum points and 50% points
    colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "k"]
    fancy_names = [r'$WI_{2006}$', r'$WI_{2015}$', r'$AWEI_{shadow}$',
                   r'$AWEI_{no shadow}$', r'$NDWI_{McFeeters}$', r'$NDWI_{Xu}$',
                   r'$TCW_{Crist}$']
    for i in range(7):
        d = D[i]
        ax.plot(d[:,3], d[:,2], ls="-", lw=1.5, color=colors[i], label=fancy_names[i])
        ax.plot(d[:,3][d[:,0] == optimum_thresholds[i]],
                d[:,2][d[:,0] == optimum_thresholds[i]], marker="o", color='k')
        ax.plot(d[:,3][d[:,0] == 50],
                d[:,2][d[:,0] == 50], marker="*", color=colors[i], markeredgecolor=colors[i], markersize=8)
    
    plt.legend(loc=(1, 0.2), fontsize=12, frameon=False, handletextpad=0.2)
    plt.savefig("best_waterindex_mixed_roc_curves.png")
    plt.savefig("figure_7.eps")
    plt.clf()
    
    # Calculate area under the curve (AUC) using the trapezoidal rule
    AUC = []
    for i in range(7):
        fpr_tpr = D[i][:,2:4]/100.0
        fpr_tpr[:,0] = np.where(fpr_tpr[:,0] == 0, 1, fpr_tpr[:,0])
        fpr_tpr = np.vstack([np.array([[0.0, 0.0]]), fpr_tpr, np.array([[1.0, 1.0]])])
        auc = metrics.auc(fpr_tpr[:, 1], fpr_tpr[:, 0], reorder=True) * 100
        AUC.append(auc)
    
    return (optimum_thresholds, D, AUC)


def calculate_mixed_statistics(mixed_thresholds, D):
    
    r = np.zeros((10, 7), dtype=float)

    # Get accuracy statistics using 50% reference thresholds.
    for i in range(7):
        d = D[i]
        (thresh, acc, TPR, FPR, users) = d[d[:, 0] == 50][0]
        r[0:5, i] = (thresh, acc, TPR, users, FPR)

    # Get accuracy statistics using optimum reference thresholds.
    for i in range(7):
        d = D[i]
        (thresh, acc, TPR, FPR, users) = d[d[:, 0] == mixed_thresholds[i]][0]
        r[5:10, i] = (thresh, acc, TPR, users, FPR)
    
    return r


def mixed_type_statistics(ref_data, index_data, thresholds, water_ads_rgb, nonwater_ads_rgb):
    
    [water_red, water_gre, water_blu] = water_ads_rgb
    [nonwater_red, nonwater_gre, nonwater_blu] = nonwater_ads_rgb
    
    # Select only mixed water pixels with ads40 RGB > 0
    s = np.where(ref_data > 100, np.where(ref_data < 200, 1, 0), 0)
    s[water_red + water_gre + water_blu <= 0] = 0
    ref_data = ref_data[s == 1] - 100
    
    # Classify on peak in green or red, or no-peak/dark
    watertype = np.where(water_gre[s == 1] >= 1.1*water_red[s == 1], "green",
                np.where(water_red[s == 1] >= 1.1*water_gre[s == 1], "brown", "other"))
    watertype = np.where(water_gre[s == 1] <= 20, np.where(water_red[s == 1] <= 20, "other", watertype), watertype)
    landtype = np.where(nonwater_gre[s == 1] >= 1.1*nonwater_red[s == 1], "vegetated",
               np.where(nonwater_red[s == 1] >= 1.1*nonwater_gre[s == 1], "bare", "other"))
    landtype = np.where(nonwater_gre[s == 1] <= 20, np.where(nonwater_red[s == 1] <= 20, "other", landtype), landtype)
    pixeltype = np.where(watertype == "brown",
                np.where(landtype == "bare", "brown-bare",
                np.where(landtype == "vegetated", "brown-vegetated", "brown-other")), "NA")
    pixeltype = np.where(watertype == "green",
                np.where(landtype == "bare", "green-bare",
                np.where(landtype == "vegetated", "green-vegetated", "green-other")), pixeltype)
    pixeltype = np.where(watertype == "other",
                np.where(landtype == "bare", "other-bare",
                np.where(landtype == "vegetated", "other-vegetated", "other-other")), pixeltype)
    
    # Make the result array
    r = np.zeros((20, 9), dtype=float)

    # First do all pixels
    for i in range(7):
        index_i = index_data[:, i]
        mixed = index_i[s == 1]
        if i == 0:
            water = ref_data[mixed <= thresholds[i]]
            nonwater = ref_data[mixed > thresholds[i]]
        else:
            water = ref_data[mixed >= thresholds[i]]
            nonwater = ref_data[mixed < thresholds[i]]
        pred = np.concatenate([np.where(water >= -999, 1, 0),
                               np.where(nonwater >= -999, -1, 0)])
        ref = np.concatenate([np.where(water >= 50, 1, -1),
                              np.where(nonwater >= 50, 1, -1)])
        (acc, TPR, FPR, users) = getErrors(ref, pred)
        
        r[0, :2] = [np.shape(water)[0], np.shape(nonwater)[0]]
        r[1, :2] = [np.shape(water)[0], np.shape(nonwater)[0]]
        r[0, i+2] = TPR
        r[1, i+2] = FPR
    
    # Then do mixed pixel types 
    types = ["brown-bare", "brown-vegetated", "brown-other",
             "green-bare", "green-vegetated", "green-other",
             "other-bare", "other-vegetated", "other-other"]
    
    for j, w in enumerate(types):
        for i in range(7):
            index_i = index_data[:, i]
            mixed = index_i[s == 1]
            mixed = mixed[pixeltype == w]
            
            if i == 0:
                water = ref_data[pixeltype == w][mixed <= thresholds[i]]
                nonwater = ref_data[pixeltype == w][mixed > thresholds[i]]
            else:
                water = ref_data[pixeltype == w][mixed >= thresholds[i]]
                nonwater = ref_data[pixeltype == w][mixed < thresholds[i]]
            
            pred = np.concatenate([np.where(water >= -999, 1, 0),
                                   np.where(nonwater >= -999, -1, 0)])
            ref = np.concatenate([np.where(water >= 50, 1, -1),
                                  np.where(nonwater >= 50, 1, -1)])
            (acc, TPR, FPR, users) = getErrors(ref, pred)
        
            r[(j+1)*2, :2] = [np.shape(water)[0], np.shape(nonwater)[0]]
            r[(j+1)*2+1, :2] = [np.shape(water)[0], np.shape(nonwater)[0]]
            r[(j+1)*2, i+2] = TPR
            r[(j+1)*2+1, i+2] = FPR
    
    return r


def regress_error(xArray, yArray, slope, intercept):
    """
    Calculates linear regression error statistics according to equations in
    Armston et al. (2009).
    """
    pred = intercept + slope*xArray
    E = yArray - pred
    RMSE = math.sqrt((1 / float(len(xArray))) * np.sum(E**2))
    meandiff = E - np.mean(E)
    variance = (1/float(len(xArray))) * np.sum(meandiff**2)
    bias = (1/float(len(xArray))) * np.sum(E)
    sser = np.sum((yArray - pred)**2)
    sstot = np.sum((yArray - np.mean(yArray))**2)
    rsquared = 1 - (sser / sstot)
    return (RMSE, variance, bias, rsquared)


def plot_level_figure(ref, indexes, thresh, plots):
    
    # Correct the scaling for the reference water percentage
    ref = np.where(ref < 100, 0, ref - 100)
    
    # Iterate through each index and classify using thresholds:
    for i in range(7):
        if i == 0:
            indexes[:, i] = np.where(indexes[:, i] <= thresh[i], 100, 0)
        else:
            indexes[:, i] = np.where(indexes[:, i] >= thresh[i], 100, 0)

    # Iterate through each plot and calculate percent water for ref and clas
    numplots = int(np.max(plots))
    result = np.zeros((numplots, 9), dtype=float)
    for n in range(1, numplots + 1):
        if ref[plots == n] != []:
            result[n-1, 0] = n
            result[n-1, 1] = np.mean(ref[plots == n])
            for i in range(7):
                result[n-1, 2+i] = np.mean(indexes[:, i][plots == n])

    # Do regression analysis
    s_all = []
    SLOPE = []
    INTERCEPT = []
    for i in range(2, 9):
        (slope, intercept, r_value, p_value, std_err) = stats.linregress(result[:, 1], result[:, i])
        (RMSE, variance, bias, rsquared) = regress_error(result[:, 1], result[:, i], slope, intercept)
        s_all.append([RMSE, rsquared])
        SLOPE.append(slope)
        INTERCEPT.append(intercept)
    
    sign = []
    for b in INTERCEPT:
        if b < 0:
            sign.append("-")
        else:
            sign.append("+")
    
    # Create scatter plot
    fig = plt.figure(1)
    fig.set_size_inches((8, 12))

    rects = [[0.15, 0.785, 0.3, 0.2],
             [0.65, 0.785, 0.3, 0.2],
             [0.15, 0.535, 0.3, 0.2],             
             [0.65, 0.535, 0.3, 0.2],
             [0.15, 0.285, 0.3, 0.2],
             [0.65, 0.285, 0.3, 0.2],
             [0.15, 0.035, 0.3, 0.2]]

    fancy_names = [r'$WI_{2006}$', r'$WI_{2015}$', r'$AWEI_{shadow}$',
                   r'$AWEI_{no shadow}$', r'$NDWI_{McFeeters}$', r'$NDWI_{Xu}$',
                   r'$TCW_{Crist}$']
    
    for i in range(7):
        ax = plt.axes(rects[i])
        ax.scatter(result[:, 1], result[:, i+2], marker="+", edgecolor="k", facecolor="none")
        ax.plot([0, 100], [0, 100], ls="-", lw=2, color="grey")
        ax.plot([0, 100], [INTERCEPT[i], SLOPE[i]*100 + INTERCEPT[i]], ls="--", lw=2, color="red")
        ax.set_xlim((0, 100))
        ax.set_ylim((0, 100))
        ax.set_ylabel(fancy_names[i]+'\nclassified water (%)', fontsize=12, horizontalalignment='center', labelpad=10)
        ax.set_xlabel('Reference water (%)', fontsize=12)
        ax.text(10, 85, 'y = %.2fx %s %.2f'%(SLOPE[i], sign[i], abs(INTERCEPT[i])), fontsize=10)
        ax.text(10, 75, 'RMSE = %.2f'%s_all[i][0], fontsize=10)
        ax.text(10, 65, r'$r^2$ = %.2f'%s_all[i][1], fontsize=10)

    plt.savefig("best_percent_water_plots.png")
    plt.savefig("figure_8.eps")
    plt.clf()
    
    
def applyODR(x, y, beta0):
    """
    Apply an ODR fit
    """
    
    def __linearfun(B, x):
        """
        Linear function y = m*x + b
        B is a vector of the parameters.
        x is an array of the current x values.
        x is same format as the x passed to Data or RealData.
        Return an array in the same format as y passed to Data or RealData.
        """
        return B[0]*x + B[1]

    # Create a Model.
    linear = odr.Model(__linearfun)
    # Create a Data instance.
    mydata = odr.Data(x, y, wd=1.0, we=1.0)
    # Instantiate ODR with your data, model and initial parameter estimate.
    myodr = odr.ODR(mydata, linear, beta0=beta0)
    # Run the fit.
    myoutput = myodr.run()
    # Derive output
    (slope, intercept) = myoutput.beta
    r2 = 1.0 - myoutput.sum_square_eps / np.sum((y - np.mean(y))**2)
    se = np.sqrt(myoutput.res_var / (y.size-1.0))
    # Calculate RMSE
    pred = intercept + slope*x
    E = y - pred
    RMSE = math.sqrt((1 / float(len(x))) * np.sum(E**2))

    return (slope, intercept, r2)


def l5_vs_l7(datafile, thresholds):
    
    # Load data
 
    l5_ref = []
    l5_b1 = []
    l5_b2 = []
    l5_b3 = []
    l5_b4 = []
    l5_b5 = []
    l5_b7 = []
    l5_wi2006 = []
    l7_ref = []
    l7_b1 = []
    l7_b2 = []
    l7_b3 = []
    l7_b4 = []
    l7_b5 = []
    l7_b7 = []
    l7_wi2006 = []
    
    with open(datafile, 'r') as f:
        for line in f:
            if line != [] and string.split(line, ' ')[0] != 'l5_ads40':
                l = string.split(line, ' ')
                l5_ref.append(int(l[3]))
                l5_wi2006.append(int(l[4]))
                l5_b1.append(float(l[5]))
                l5_b2.append(float(l[6]))
                l5_b3.append(float(l[7]))
                l5_b4.append(float(l[8]))
                l5_b5.append(float(l[9]))
                l5_b7.append(float(l[10]))
                l7_ref.append(int(l[14]))
                l7_wi2006.append(int(l[15]))
                l7_b1.append(float(l[16]))
                l7_b2.append(float(l[17]))
                l7_b3.append(float(l[18]))
                l7_b4.append(float(l[19]))
                l7_b5.append(float(l[20]))
                l7_b7.append(float(l[21]))
                
    l5_ref = np.asarray(l5_ref).astype(np.float32)
    l5_wi2006 = np.asarray(l5_wi2006).astype(np.float32)
    l5_b1 = np.asarray(l5_b1).astype(np.float32)
    l5_b2 = np.asarray(l5_b2).astype(np.float32)
    l5_b3 = np.asarray(l5_b3).astype(np.float32)
    l5_b4 = np.asarray(l5_b4).astype(np.float32)
    l5_b5 = np.asarray(l5_b5).astype(np.float32)
    l5_b7 = np.asarray(l5_b7).astype(np.float32)
    l7_ref = np.asarray(l7_ref).astype(np.float32)
    l7_wi2006 = np.asarray(l7_wi2006).astype(np.float32)
    l7_b1 = np.asarray(l7_b1).astype(np.float32)
    l7_b2 = np.asarray(l7_b2).astype(np.float32)
    l7_b3 = np.asarray(l7_b3).astype(np.float32)
    l7_b4 = np.asarray(l7_b4).astype(np.float32)
    l7_b5 = np.asarray(l7_b5).astype(np.float32)
    l7_b7 = np.asarray(l7_b7).astype(np.float32)
    l5_dbg = np.transpose(np.vstack([l5_b1, l5_b2, l5_b3, l5_b4, l5_b5, l5_b7])) * 10000
    l7_dbg = np.transpose(np.vstack([l7_b1, l7_b2, l7_b3, l7_b4, l7_b5, l7_b7])) * 10000
    
    # Select where not null
    null = np.where(l5_b1 > 1, 1, 0)
    null[l5_b2 > 1] = 1
    null[l5_b3 > 1] = 1
    null[l5_b4 > 1] = 1
    null[l5_b5 > 1] = 1
    null[l5_b7 > 1] = 1
    null[l7_b1 > 1] = 1
    null[l7_b2 > 1] = 1
    null[l7_b3 > 1] = 1
    null[l7_b4 > 1] = 1
    null[l7_b5 > 1] = 1
    null[l7_b7 > 1] = 1
    l5_ref = l5_ref[null == 0]
    l7_ref = l7_ref[null == 0]
    l5_wi2006 = l5_wi2006[null == 0]
    l7_wi2006 = l7_wi2006[null == 0]
    l5_dbg = l5_dbg[null == 0, :]
    l7_dbg = l7_dbg[null == 0, :]
    
    # Select data where l5_ref and l7_ref values are both 100% water
    # Include all pure non-water as well  
    pure_water_selection = np.where(l5_ref == 200, np.where(l7_ref == 200, 1, 0), 0)
    pure_nonwater_selection = np.where(l5_ref <= 100, np.where(l7_ref <= 100, 1, 0), 0)
    pure = pure_water_selection + pure_nonwater_selection
    
    # Calculate index values
    (names, l5_indexes) = calculate_models(l5_wi2006, None, l5_dbg)
    (names, l7_indexes) = calculate_models(l7_wi2006, None, l7_dbg)
    
    # Calculate accuracy
    r = np.zeros((8, 7), dtype=float)
    comp = np.zeros((4, 7), dtype=float)
    
    for i in range(7):
        t = thresholds[i]
        
        # l5 accuracy        
        index_i = l5_indexes[:,i]
        water = index_i[pure_water_selection == 1]
        nonwater = index_i[pure_nonwater_selection == 1]
        ref = np.concatenate([np.where(water >= -999, 1, 0),
                              np.where(nonwater >= -999, -1, 0)])
        if i == 0:
              pred = np.concatenate([np.where(water <= thresholds[i], 1, -1),
                                     np.where(nonwater <= thresholds[i], 1, -1)])      
        else:
              pred = np.concatenate([np.where(water >= thresholds[i], 1, -1),
                                     np.where(nonwater >= thresholds[i], 1, -1)])
        (acc, prods, fpr, users) = getErrors(ref, pred)
        r[0:4, i] = [acc, prods, users, fpr]

        # l7 accuracy
        index_i = l7_indexes[:,i]
        water = index_i[pure_water_selection == 1]
        nonwater = index_i[pure_nonwater_selection == 1]
        ref = np.concatenate([np.where(water >= -999, 1, 0),
                              np.where(nonwater >= -999, -1, 0)])
        if i == 0:
              pred = np.concatenate([np.where(water <= thresholds[i], 1, -1),
                                     np.where(nonwater <= thresholds[i], 1, -1)])      
        else:
              pred = np.concatenate([np.where(water >= thresholds[i], 1, -1),
                                     np.where(nonwater >= thresholds[i], 1, -1)])
        (acc, prods, fpr, users) = getErrors(ref, pred)
        r[4:, i] = [acc, prods, users, fpr]
        
        # Comparison
        for i in range(7):
            l7_i = l7_indexes[:,i]
            l5_i = l5_indexes[:,i]
            if i == 0:
                ref = np.where(l7_i <= thresholds[i], 1, -1)
                pred = np.where(l5_i <= thresholds[i], 1, -1)     
            else:
                ref = np.where(l7_i >= thresholds[i], 1, -1)
                pred = np.where(l5_i >= thresholds[i], 1, -1)
              
            a = np.sum(np.where(ref == -1, 0, np.where(pred == -1, 0, 1)))
            b = np.sum(np.where(ref ==  1, 0, np.where(pred == -1, 0, 1)))
            c = np.sum(np.where(ref == -1, 0, np.where(pred ==  1, 0, 1)))
            d = np.sum(np.where(ref ==  1, 0, np.where(pred ==  1, 0, 1)))
            comp[0:4, i] = [a, b, c, d]
    
    # Write comparison
    output = "/mnt/project/landsat_water/scripts/l7_l5_waterindex_comparison.txt"
    with open(output, "w") as f:
        f.write("\nETM+ and TM accuracy comparison\n\n")
        for i in range(7):
            line = "%i\t%i\t%i\t%i\n"%(comp[0, i], comp[1, i], comp[2, i], comp[3, i])
            f.write(line)
    
    # Calculate regression statistics using ODR
    SLOPE = []
    INTERCEPT = []
    RSQUARED = []
    for i in range(7):
        (slope, intercept, r2) = applyODR(l5_indexes[:, i][pure == 1], l7_indexes[:, i][pure == 1], [1, 0])
        SLOPE.append(slope)
        INTERCEPT.append(intercept)
        RSQUARED.append(r2)
    
    sign = []
    for b in INTERCEPT:
        if b < 0:
            sign.append("-")
        else:
            sign.append("+")
    
    # Create density plot of l5 vs l7 index values
    fig = plt.figure(1)
    fig.set_size_inches((8, 12))
    rects = [[0.15, 0.785, 0.3, 0.2],
             [0.55, 0.785, 0.3, 0.2],
             [0.15, 0.535, 0.3, 0.2],
             [0.55, 0.535, 0.3, 0.2],
             [0.15, 0.285, 0.3, 0.2],
             [0.55, 0.285, 0.3, 0.2],
             [0.15, 0.035, 0.3, 0.2]]
    
    fancy_names = [r'$WI_{2006}$', r'$WI_{2015}$', r'$AWEI_{shadow}$',
                   r'$AWEI_{no shadow}$', r'$NDWI_{McFeeters}$', r'$NDWI_{Xu}$',
                   r'$TCW_{Crist}$']
    minmax = [[40, 100], [-50, 50], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-0.25, 0.25]]
    
    textpos = [[0.165, 0.96], [0.565, 0.96], [0.165, 0.71], [0.565, 0.71],
               [0.165, 0.46], [0.565, 0.46], [0.165, 0.21]]

    for i in range(7):
        ax = plt.axes(rects[i])
        width = (minmax[i][1] - minmax[i][0]) / 50.0
        xbins = np.concatenate((np.arange(minmax[i][0], minmax[i][1], width, dtype=float), [minmax[i][1]]))
        ybins = np.concatenate((np.arange(minmax[i][0], minmax[i][1], width, dtype=float), [minmax[i][1]]))
        xbins = xbins[0:51] 
        ybins = ybins[0:51]    
        (density, yedges, xedges) = np.histogram2d(l7_indexes[:, i], l5_indexes[:, i], bins=(ybins, xbins))
        density = np.flipud(density)
        den_nozero = np.where(density==0, 1, density)
        density = np.where(density==0, 0, np.log10(den_nozero))
        density = np.ma.masked_where((density==0), density)
        cmap = CM.get_cmap('jet')
        cmap.set_bad('w')
        extent = [minmax[i][0], minmax[i][1], minmax[i][0], minmax[i][1]]
        im = ax.imshow(density, extent=extent, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.plot([-100, 100], [-100, 100], ls="-", lw=1, color='k')
        ax.plot([-100, 100], [-100*SLOPE[i] + INTERCEPT[i], 100*SLOPE[i] + INTERCEPT[i]], ls="--", lw=1.5, color="0.7")
        ax.plot([-100, 100], [thresholds[i], thresholds[i]], ls=":", lw=1.5, color="k")
        ax.plot([thresholds[i], thresholds[i]], [-100, 100], ls=":", lw=1.5, color="k")
        ax.set_ylabel('ETM+ water index', fontsize=12)
        ax.set_xlabel('TM water index', fontsize=12)
        ax.set_xlim((minmax[i][0], minmax[i][1]))
        ax.set_ylim((minmax[i][0], minmax[i][1]))
        fig.text(textpos[i][0], textpos[i][1], fancy_names[i], fontsize=12, fontweight='bold')
        fig.text(textpos[i][0], textpos[i][1] - 0.02, r'$y = %.2fx %s %.2f$'%(SLOPE[i], sign[i], abs(INTERCEPT[i])), fontsize=12)
        fig.text(textpos[i][0], textpos[i][1] - 0.04, r'$r^2 = %.2f$'%RSQUARED[i], fontsize=12)

    # Add colorbar
    rect_cbar = [0.62, 0.14, 0.15, 0.01]
    cax  = plt.axes(rect_cbar)
    cbar = fig.colorbar(im, cax=cax, ticks=[np.min(density), np.max(density)],
                        orientation='horizontal')
    cbar.ax.set_xticklabels(['minimum', 'maximum'], fontsize=10)
    plt.figtext(0.625, 0.157, 'Relative density', fontsize=10)
    plt.savefig("l5_vs_l7_water_indexes.png")
    plt.clf()
    
    return r


# Just hardcode the sequence

# Load data
validation_data = "/mnt/project/landsat_water/validation/reference_data_wi2015.txt"
(ads40, ref, WI2006, dbf, dbg, water_ads_rgb, nonwater_ads_rgb, plots) = load_data(validation_data)

# Create equation values
(names, models) = calculate_models(WI2006, dbf, dbg)

# Calculate optimum thresholds for pure pixels
(optimum_thresholds, pure_AUC) = ROC_optimum(ref, models, names)

#optimum_thresholds = [72.994999999998811, 0.6283836, -0.02471004, -0.06955683, -0.21188074, 0.00463429, -0.00827649]
#optimum_thresholds = [68, 0.63, -0.02, -0.07, -0.21, 0, -0.01]

# Calculate accuracy statistics using optimum thresholds
pure_stats = calculate_statistics(ref, models, optimum_thresholds)

# Calculate operational accuracy
operational_stats = calculate_operational_statistics(ref, models, optimum_thresholds, ads40, water_ads_rgb[0])

# Calculate omission and commission errors for classes (water color and land type)
(omi, com) = calculate_errors(ads40, water_ads_rgb[0], ref, dbg, models, optimum_thresholds)

# Create histograms of the indexes separated into water/non-water
make_histograms(ref, models, optimum_thresholds)

# Create histograms of the indexes separated into the clear-water and coloured water
coloured_histograms(ref, dbg, models, optimum_thresholds)

# Calculate optimum thresholds for reference water content of mixed pixels
(mixed_thresholds, mixed_stats, mixed_AUC) = mixed_ROC_optimum(ref, models, optimum_thresholds)

# Calculate accuracy statistics for mixed pixels
mixed_stats = calculate_mixed_statistics(mixed_thresholds, mixed_stats)

# Calculate accuracy statistics for mixed pixels by water types
mixed_type_stats = mixed_type_statistics(ref, models, optimum_thresholds, water_ads_rgb, nonwater_ads_rgb)

# Plot level assessment
plot_level_figure(ref, models, optimum_thresholds, plots)

# TM vs ETM+ assessment
l57_data = "/mnt/project/landsat_water/validation/wi2015_l5-7_reference_data.txt"
l57_accuracy = l5_vs_l7(l57_data, optimum_thresholds)

# Save optimum thresholds and accuracy statistics for all models 
output = "/mnt/project/landsat_water/scripts/best_waterindex_comparison.txt"
with open(output, "w") as f:
    
    # Pure pixel accuracy
    f.write("Pure pixel accuracy (Table 5)\n\n")
    line = "\tPixels"
    for n in names:
        line = "%s\t%s"%(line, n)
    line = "%s\n"%line
    f.write(line)
    line = "Area under the ROC curve\t%i"%(np.sum(omi[:,0]) + np.sum(com[:,0]))
    for i in range(7):
        line = "%s\t%.1f"%(line, pure_AUC[i])
    line = "%s\n"%line
    f.write(line)
    line = "Optimum index threshold\t%i"%(np.sum(omi[:,0]) + np.sum(com[:,0]))
    for i in range(7):
        line = "%s\t%.2f"%(line, pure_stats[i, 0])
    line = "%s\n"%line
    f.write(line)
    line = "Overall accuracy\t%i"%(np.sum(omi[:,0]) + np.sum(com[:,0]))
    for i in range(7):
        line = "%s\t%.1f"%(line, pure_stats[i, 1])
    line = "%s\n"%line
    f.write(line)
    line = "Water producers' accuracy\t%i"%(np.sum(omi[:,0]) + np.sum(com[:,0]))
    for i in range(7):
        line = "%s\t%.1f"%(line, pure_stats[i, 2])
    line = "%s\n"%line
    f.write(line)
    line = "Water users accuracy\t%i"%(np.sum(omi[:,0]) + np.sum(com[:,0]))
    for i in range(7):
        line = "%s\t%.1f"%(line, pure_stats[i, 3])
    line = "%s\n"%line
    f.write(line)
    line = "False positive rate\t%i"%(np.sum(omi[:,0]) + np.sum(com[:,0]))
    for i in range(7):
        line = "%s\t%.1f"%(line, pure_stats[i, 4])
    line = "%s\n"%line
    f.write(line)

    # Pure pixel omission errors
    watertypes = ["ocean", "clear-deep", "dark-green", "green", "green-brown", "brown", "dark-brown"]
    for i in range(7):
        line = "%s\t%i\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\n"%(watertypes[i], omi[i,0], omi[i,1], omi[i,2],
                                                                omi[i,3], omi[i,4], omi[i,5], omi[i,6], omi[i,7])
        f.write(line)
        
    # Pure pixel commission errors
    landtypes = ['bare ground', 'sparse vegetation', 'dense vegetation', 'cloud-shadow', 'topographic-shadow', 'quarries', 'urban']
    for i in range(7):
        line = "%s\t%i\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\n"%(landtypes[i], com[i,0], com[i,1], com[i,2],
                                                               com[i,3], com[i,4], com[i,5], com[i,6], com[i,7])
        f.write(line)
        
    # Mixed pixel accuracy
    line = "Area under the ROC curve\t"
    for i in range(7):
        line = "%s\t%.1f"%(line, mixed_AUC[i])
    line = "%s\n"%line
    f.write(line)
    line = "Reference threshold"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[0, i])
    line = "%s\n"%line
    f.write(line)
    line = "Overall accuracy"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[1, i])
    line = "%s\n"%line
    f.write(line)
    line = "Water producers' accuracy"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[2, i])
    line = "%s\n"%line
    f.write(line)
    line = "Water users' accuracy"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[3, i])
    line = "%s\n"%line
    f.write(line)
    line = "False positive rate"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[4, i])
    line = "%s\n"%line
    f.write(line)
    line = "Optimum reference threshold"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[5, i])
    line = "%s\n"%line
    f.write(line)
    line = "Overall accuracy"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[6, i])
    line = "%s\n"%line
    f.write(line)
    line = "Water producers' accuracy*"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[7, i])
    line = "%s\n"%line
    f.write(line)
    line = "Water users' accuracy"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[8, i])
    line = "%s\n"%line
    f.write(line)
    line = "False positive rate"
    for i in range(7):
        line = "%s\t%.0f"%(line, mixed_stats[9, i])
    line = "%s\n"%line
    f.write(line)
    
    # Mixed type accuracy
    f.write("\nMixed pixel accuracy by water type (Table 7)\n\n")
    (rows, cols) = np.shape(mixed_type_stats)
    for r in range(rows):
        line = "%.0f"%mixed_type_stats[r, 0]
        for c in range(1, cols):
            line = "%s\t%.0f"%(line, mixed_type_stats[r, c])
        line = "%s\n"%line
        f.write(line)
        if r in range(1, 20, 2):
            f.write("\n")

    # TM and ETM+ accuracy comparison
    f.write("\nTM and ETM+ accuracy comparison (Table 8)\n\n")
    (rows, cols) = np.shape(l57_accuracy)
    for r in range(rows):
        line = "%.1f"%l57_accuracy[r, 0]
        for c in range(1, cols):
            line = "%s\t%.1f"%(line, l57_accuracy[r, c])
        line = "%s\n"%line
        f.write(line)
        if r == 3:
            f.write("\n")
    
    # Operational accuracy 
    f.write("\nOperational accuracy\n\n")
    for j in range(4):
        line = ""
        for i in range(7):
            line = "%s\t%.1f"%(line, operational_stats[i, j])
        line = "%s\n"%line
        f.write(line)
