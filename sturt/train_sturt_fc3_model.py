#!/usr/bin/env python
"""
Needs the fc3 conda environment
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskElasticNet


params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'xtick.direction': 'out', 'ytick.direction': 'out'}
plt.rcParams.update(params)

# Compare drone and FC3 model
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\comparison_fc_v3.csv'
fc3_id = []
fc3_site = []
fc3_date = []
drone = []
fc3 = []
with open(csvfile, 'r') as f:
    f.readline()
    for line in f:
        fc3_id.append(int(line.split(',')[0]))
        fc3_site.append(line.split(',')[1])
        fc3_date.append(float(line.split(',')[2]))
        living = float(line.split(',')[3])
        dead = float(line.split(',')[4])
        bare = 100 - (living + dead)
        drone.append([living, dead, bare])
        pv = float(line.split(',')[5])
        npv = float(line.split(',')[6])
        bs = float(line.split(',')[7])  
        fc3.append([pv, npv, bs])
fc3_id = np.array(fc3_id)
fc3_site = np.array(fc3_site)
fc3_date = np.array(fc3_date)
droneCover = np.array(drone, dtype=np.float32)
fc3Cover = np.array(fc3, dtype=np.float32)

rmse_living = np.sqrt(np.mean((droneCover[:, 0] - fc3Cover[:, 0])**2))
rmse_dead = np.sqrt(np.mean((droneCover[:, 1] - fc3Cover[:, 1])**2))
rmse_bare = np.sqrt(np.mean((droneCover[:, 2] - fc3Cover[:, 2])**2))
print("FC3 RMSE") 
print("Living", rmse_living)
print("Dead", rmse_dead)
print("Bare", rmse_bare)
print("Mean sum of fractions", np.mean(np.sum(fc3Cover, axis=1)))

# Make plot
fig = plt.figure(1)
fig.set_size_inches((6, 2))
rects  = [[0.15, 0.25, 0.2, 0.6], [0.45, 0.25, 0.2, 0.6], [0.75, 0.25, 0.2, 0.6]]
for x in range(3):
    ax = plt.axes(rects[x])
    drone = droneCover[:, x]
    satellite = fc3Cover[:, x] 
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    if x == 0:
        ax.set_ylabel('\nLandsat cover (%)', fontsize=10)
    if x == 1:
        ax.set_xlabel('Drone cover (%)', fontsize=10)
    ax.plot(drone, satellite, ls='', marker='.', markeredgecolor='0.5', markerfacecolor='None')
    ax.plot([0, 100], [0, 100], ls='-', color='k', lw=1)

fig.text(0.15, 0.9, 'Living', fontsize=10)
fig.text(0.45, 0.9, 'Dead', fontsize=10)
fig.text(0.75, 0.9, 'Bare', fontsize=10)
plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\landsat_fc3_model.png', dpi=300)
plt.clf()


# Get the FC3 model
def get_model(n_inputs, n_outputs):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            20, input_dim=n_inputs, kernel_initializer="he_uniform", activation="relu"
        )
    )
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss="mae", optimizer="adam")
    return model


# Read in cal-val data
# train_dataset = [green, red, nir, swir1, swir2]
# train_labels = [bare, living, dead]
csvfile = r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\sturt_calval.csv'
date = []
Id = []
site = []
train_dataset = []
train_labels = []
with open(csvfile, 'r') as f:
    f.readline()
    for line in f:
        # id,site,date,living,dead,bare,b1,b2,b3,b4,b5,b6
        Id.append(int(line.split(',')[0]))
        site.append(line.split(',')[1])
        year = float(line.split(',')[2][:4])
        date.append(year + (4.5/12.0))
        living = float(line.split(',')[3])
        dead = float(line.split(',')[4])
        bare = float(line.split(',')[5])
        train_labels.append([living, dead, bare])
        bands = [float(b) for b in line.split(',')[6:12]]
        train_dataset.append(bands)
Id = np.array(Id)
site = np.array(site)
date = np.array(date)
train_dataset = np.array(train_dataset, dtype=np.float32) / 10000.0 # Convert to reflectance (0-1)
train_labels = np.array(train_labels, dtype=np.float32) / 100.0 # Convert to proportions (0-1)

# Add multiplicative predictors
x = train_dataset
n = x.shape[1]
for i in range(0, n):
    for j in range(i, n):
        ijPred = np.transpose(np.array([x[:, i] * x[:, j]]))
        x = np.hstack((x, ijPred))

# Centre and normalise x
x_centered = x - np.mean(x, axis=0)
x_normalized = (x_centered / np.sqrt(np.sum((x_centered)**2, axis=0)))

# Centre y
y = train_labels
constant = np.mean(y, axis=0)
y_centered = y - constant

####################
# Tensorflow model #
####################

# Fit model
model = get_model(27, 3)
model.fit(x_normalized, y_centered, verbose=0, epochs=100)

# Get drone and satellite cover as measured and modelled
d = train_labels * 100.0
s = model.predict(x_normalized)
s[:, 0] = s[:, 0] + constant[0]
s[:, 1] = s[:, 1] + constant[1]
s[:, 2] = s[:, 2] + constant[2]
s = s * 100.0
s[s < 0] = 0
s[s > 100] = 100

tensor = s

# Make plot
fig = plt.figure(1)
fig.set_size_inches((6, 2))
rects  = [[0.15, 0.25, 0.2, 0.6], [0.45, 0.25, 0.2, 0.6], [0.75, 0.25, 0.2, 0.6]]
for x in range(3):
    ax = plt.axes(rects[x])
    drone = d[:, x]
    satellite = s[:, x] 
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    if x == 0:
        ax.set_ylabel('\nLandsat cover (%)', fontsize=10)
    if x == 1:
        ax.set_xlabel('Drone cover (%)', fontsize=10)
    ax.plot(drone, satellite, ls='', marker='.', markeredgecolor='0.5', markerfacecolor='None')
    ax.plot([0, 100], [0, 100], ls='-', color='k', lw=1)

fig.text(0.15, 0.9, 'Living', fontsize=10)
fig.text(0.45, 0.9, 'Dead', fontsize=10)
fig.text(0.75, 0.9, 'Bare', fontsize=10)
plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\landsat_tensorflow_model.png', dpi=300)
plt.clf()

# Assess accuracy with leave-one-out cross valiation
result = np.zeros_like(y_centered)
n = np.shape(x_normalized)[0]
for i in range(n):
    x = np.delete(x_normalized, i, axis=0)
    y = np.delete(y_centered, i, axis=0)
    m = get_model(27, 3)
    m.fit(x, y, verbose=0, epochs=100)
    p = m.predict(np.expand_dims(x_normalized[i, :], axis=0))[0]
    result[i, :] = p

result[:, 0] = result[:, 0] + constant[0]
result[:, 1] = result[:, 1] + constant[1]
result[:, 2] = result[:, 2] + constant[2]
result = result * 100.0
result[result < 0] = 0
result[result > 100] = 100

rmse_living = np.sqrt(np.mean((d[:, 0] - result[:, 0])**2))
rmse_dead = np.sqrt(np.mean((d[:, 1] - result[:, 1])**2))
rmse_bare = np.sqrt(np.mean((d[:, 2] - result[:, 2])**2))
print("Tensorflow RMSE") 
print("Living", rmse_living)
print("Dead", rmse_dead)
print("Bare", rmse_bare)
print("Mean sum of fractions", np.mean(np.sum(result, axis=1)))

################################
# Elastic-net regression model #
################################

# Add multiplicative predictors
x = train_dataset
n = x.shape[1]
for i in range(0, n):
    for j in range(i, n):
        ijPred = np.transpose(np.array([x[:, i] * x[:, j]]))
        x = np.hstack((x, ijPred))

# Centre and normalise x
x_centered = x - np.mean(x, axis=0)
x_normalized = (x_centered / np.sqrt(np.sum((x_centered)**2, axis=0)))

# Centre y
y = train_labels
constant = np.mean(y, axis=0)
y_centered = y - constant

# Elastic net regression with 5-fold cv to get the best parameters
regr = MultiTaskElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                             fit_intercept=False, cv=5, max_iter=1000000)
regr.fit(x_normalized, y_centered)
best_l1_ratio = regr.l1_ratio_
best_alpha = regr.alpha_
coefficients = regr.coef_

# Elastic net regression with LOOCV to calculate RMSE
regr = MultiTaskElasticNet(l1_ratio=best_l1_ratio, alpha=best_alpha,
                           fit_intercept=False, max_iter=1000000)
model = np.zeros_like(y)
k = np.arange(model.shape[0])
for j in k:
    xj = x_normalized[k != j, :]
    yj = y_centered[k != j]
    regr.fit(xj, yj)
    coefficients = regr.coef_
    for i in range(3):
        model[k == j, i] = ((np.sum(coefficients[i, :] *
                                    x_normalized[(k == j), :]) +
                                    constant[i]))
model = model * 100.0
model[model < 0] = 0
model[model > 100] = 100

measu = train_labels * 100.0
print("Elastic net model RMSE")
iName = ["Living", "Dead", "Bare"]
for i in range(3):
    iModel = model[:, i]
    iMeasu = measu[:, i]
    iRmse = np.sqrt(np.mean((iModel - iMeasu)**2))
    print(iName[i], iRmse)
print("Mean sum of fractions", np.mean(np.sum(model, axis=1)))

# Elastic net regression with all data to get best model coefficients
regr.fit(x_normalized, y_centered)
coefficients = regr.coef_
model = np.zeros_like(y)
for i in range(3):
    model[:, i] = ((np.sum(coefficients[i, :] * x_normalized, axis=1) + constant[i]))

elastic = model * 100.0
elastic[elastic < 0] = 0
elastic[elastic > 100] = 100

# Save all data needed to implement model
# - scaling of inputs needs mean and standard deviation of each predictor
# - scaling of outputs needs the constant (mean of drone values)
# - calculation of model needs coefficients
print('\nMean of input bands')
print(np.mean(x, axis=0))
print('\nStandard deviation of normalised bands')
print(np.sqrt(np.sum((x_centered)**2, axis=0)))
print('\nConstant')
print(constant)
print('\nCoefficients')
print(coefficients)

# Make plot
fig = plt.figure(1)
fig.set_size_inches((6, 2))
rects  = [[0.15, 0.25, 0.2, 0.6], [0.45, 0.25, 0.2, 0.6], [0.75, 0.25, 0.2, 0.6]]
for x in range(3):
    ax = plt.axes(rects[x])
    drone = train_labels[:, x] * 100.0
    satellite = model[:, x] * 100.0
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    if x == 0:
        ax.set_ylabel('\nLandsat cover (%)', fontsize=10)
    if x == 1:
        ax.set_xlabel('Drone cover (%)', fontsize=10)
    ax.plot(drone, satellite, ls='', marker='.', markeredgecolor='0.5', markerfacecolor='None')
    ax.plot([0, 100], [0, 100], ls='-', color='k', lw=1)

fig.text(0.15, 0.9, 'Living', fontsize=10)
fig.text(0.45, 0.9, 'Dead', fontsize=10)
fig.text(0.75, 0.9, 'Bare', fontsize=10)
plt.savefig(r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\landsat_elasticnet_model.png', dpi=300)
plt.clf()

# Save all the models as a CSV file
drone = train_labels * 100.0
b = train_dataset * 10000

fc3 = np.zeros_like(fc3Cover)
for i in range(Id.size):
    position = [(fc3_id == Id[i]) & (fc3_site == site[i]) & (fc3_date == date[i])][0]
    fc3[i, :] = fc3Cover[position, :]

header = ('id,site,date,droneLiving,droneDead,droneBare,band1,band2,band3,' +
          'band4,band5,band6,fc3Living,fc3Dead,fc3Bare,tensorLiving,' +
          'tensorDead,tensorBare,elasticLiving,elasticDead,elasticBare\n')

with open(r'C:\Users\Adrian\OneDrive - UNSW\Documents\papers\preparation\wild_deserts_vegetation_change\landsat_models.csv', 'w') as f:
    f.write(header) 
    for i in range(Id.size):
        line = '%i,%s,%.3f'%(Id[i], site[i], date[i])
        line = '%s,%.3f,%.3f,%.3f'%(line, drone[i, 0], drone[i, 1], drone[i, 2])
        line = '%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f'%(line, b[i, 0], b[i, 1], b[i, 2], b[i, 3], b[i, 4], b[i, 5]) 
        line = '%s,%.3f,%.3f,%.3f'%(line, fc3[i, 0], fc3[i, 1], fc3[i, 2])
        line = '%s,%.3f,%.3f,%.3f'%(line, tensor[i, 0], tensor[i, 1], tensor[i, 2])
        line = '%s,%.3f,%.3f,%.3f\n'%(line, elastic[i, 0], elastic[i, 1], elastic[i, 2])
        f.write(line)