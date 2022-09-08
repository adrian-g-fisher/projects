#!/usr/bin/env python

import os
import sys
import numpy as np

csvFile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\Fowlers_Gap\land_classes\Fowlers_DSE_Shrubs_July22_2022.csv'
land_classes = []
dse_data = []   # 1981, 2022, percent change
shrub_data = [] # 1981, 2022, normalised change
with open(csvFile, 'r') as f:
    f.readline()
    for line in f:
        l = line.strip().split(',')
        land_classes.append(l[1])
        dse_data.append([float(l[4]), float(l[5]), float(l[8])])
        shrub_data.append([float(l[6]), float(l[7]), float(l[11])])
land_classes = np.array(land_classes)
dse_data = np.array(dse_data)
shrub_data = np.array(shrub_data)

output = r'C:\Users\Adrian\OneDrive - UNSW\Documents\Fowlers_Gap\land_classes\dse_shrubs_mean.csv'
with open(output, 'w') as f:
    f.write('land_class,mean_dse_1981,mean_dse_2022,dse_change_percent,mean_shrubs_1981,mean_shrubs_2022,shrub_change_normal\n')
    
for lc in np.unique(land_classes):
    mean_dse_1981 = np.mean(dse_data[:, 0][land_classes == lc])
    mean_dse_2022 = np.mean(dse_data[:, 1][land_classes == lc])
    dse_change_percent = np.mean(dse_data[:, 2][land_classes == lc])

    mean_shrubs_1981 = np.mean(shrub_data[:, 0][land_classes == lc])
    mean_shrubs_2022 = np.mean(shrub_data[:, 1][land_classes == lc])
    shrub_change_normal = np.mean(shrub_data[:, 2][land_classes == lc])

    with open(output, 'a') as f:
        f.write('%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(lc, mean_dse_1981, mean_dse_2022, dse_change_percent,
                                                      mean_shrubs_1981, mean_shrubs_2022, shrub_change_normal))