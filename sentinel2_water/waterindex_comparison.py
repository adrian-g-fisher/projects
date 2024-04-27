#!/usr/bin/env python

"""

Adapted from Landsat water index analysis

Works in the water conda environment:
conda create -n water scipy matplotlib scikit-learn

 This does the following:
 - reads in the reflectance values for the validation data
 - calculates the water index values for each validation pixel
 - determines the optimum threshold to classify water and non-water using ROC
   curves
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
    
    return (ref, reflect, plot, indexes)


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
        water = index_i[ref_data == 100]
        nonwater = index_i[ref_data == 0]
        
        # Subset all water and non-water values
        #water = index_i[ref_data > 50]
        #nonwater = index_i[ref_data <= 50]
        
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
    #ax.set_xlim((-0.5, 6.5))
    #ax.set_ylim((93.5, 100.5))
    ax.set_xlabel('False positive rate (%)', fontsize=12)
    ax.set_ylabel('True positive rate (%)', fontsize=12)
    
    # Plot curves and optimum points
    colors = ["palegreen", "coral", "deepskyblue", "orchid"]
    fancy_names = [r'$MNDWI_{Xu}$', r'$MuWIC$', r'$TWI$', r'$WI_{Fisher}$']
    for i in range(4):
        d = D[i]
        ax.plot(d[:,3], d[:,2], ls="-", lw=1.5, color=colors[i], label=fancy_names[i])
        ax.plot(d[:,3][d[:,0] == optimum_thresholds[i]],
                d[:,2][d[:,0] == optimum_thresholds[i]], marker="o", color='k')
    
    plt.legend(loc=(1, 0.2), fontsize=12, frameon=False, handletextpad=0.2)
    plt.savefig("waterindex_roc_curves_purepixels.png")
    #plt.savefig("waterindex_roc_curves_allpixels.png")
    plt.clf()
    
    # Calculate area under the curve (AUC) using the trapezoidal rule
    AUC = []
    for i in range(4):
        fpr_tpr = D[i][:,2:4]/100.0
        fpr_tpr = np.flipud(fpr_tpr)
        fpr_tpr = np.vstack([np.array([[0.0, 0.0]]), fpr_tpr, np.array([[1.0, 1.0]])])
        row_diff = np.diff(fpr_tpr, axis=0)
        unique_rows = np.ones(len(fpr_tpr), dtype='bool')
        unique_rows[1:] = (row_diff != 0).any(axis=1) 
        fpr_tpr = fpr_tpr[unique_rows]
        auc = metrics.auc(fpr_tpr[:, 1], fpr_tpr[:, 0]) * 100
        AUC.append(auc)
    
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
        water = index_i[ref_data == 100]
        nonwater = index_i[ref_data == 0]
        
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


def make_histograms(ref, index_data, thresh):

    fancy_names = [r'$MNDWI_{Xu}$', r'$MuWIC$', r'$TWI$', r'$WI_{Fisher}$']

    rectangles = [[0.06, 0.912, 0.89, 0.1],
                  [0.06, 0.770, 0.89, 0.1],
                  [0.06, 0.628, 0.89, 0.1],
                  [0.06, 0.486, 0.89, 0.1]]

    ranges = [[-1, 1], [-1, 1], [-1, 1], [-0.25, 0.1]]

    fig = plt.figure(1)
    fig.set_size_inches((5, 8))

    for i in range(4):
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
    plt.clf()


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
            print(i, t)
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

# Just hardcode the sequence

# Load data
validation_data = "reference_data.csv"
(reference, reflectance, plots, indexes) = load_data(validation_data)

# Create equation values
(names, models) = calculate_models(reflectance, indexes)

# Calculate optimum thresholds for pure pixels
(optimum_thresholds, pure_AUC) = ROC_optimum(reference, models, names)

# Calculate accuracy statistics using optimum thresholds
pure_stats = calculate_statistics(reference, models, optimum_thresholds)

# Create histograms of the indexes separated into water/non-water
make_histograms(reference, models, optimum_thresholds)

sys.exit()

##########


# Calculate optimum thresholds for reference water content of mixed pixels
(mixed_thresholds, mixed_stats, mixed_AUC) = mixed_ROC_optimum(reference, models, optimum_thresholds)

# Calculate accuracy statistics for mixed pixels
mixed_stats = calculate_mixed_statistics(mixed_thresholds, mixed_stats)

# Plot level assessment
plot_level_figure(reference, models, optimum_thresholds, plots)

# Save optimum thresholds and accuracy statistics for all models 
output = "waterindex_comparison.txt"
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