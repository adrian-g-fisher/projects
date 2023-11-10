#!/usr/bin/env python
"""

"""

import os
import sys
import glob
import joblib
import numpy as np
from osgeo import gdal, ogr, osr
from rios import applier, rat
from sklearn.ensemble import RandomForestClassifier


def getPixelValues(info, inputs, outputs, otherargs):
    """
    Called from RIOS to extracts pixel values from point locations.
    """
    points = inputs.points[0]
    pointsPresent = np.unique(points[points != 0])
    if len(pointsPresent) > 0:
        ids = points[points != 0]
        for i in range(ids.size):
            stats = np.ndarray.flatten(inputs.ts_image[:, points == ids[i]])
            otherargs.pixels[ids[i]-1, :] = stats


# Get the attributes from the shapefile
pointfile = "data\landcover_points.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(pointfile, 0)
layer = dataSource.GetLayer()
pointIds = []
pointClasses = []
fields = ['Id', 'class']
for feature in layer:
    pointIds.append(int(feature.GetField("Id")))
    pointClasses.append(feature.GetField("class"))
pointIds = np.array(pointIds)
pointClasses = np.array(pointClasses)

# Get the pixel values from the time series image
infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()
otherargs = applier.OtherInputs()
controls = applier.ApplierControls()
controls.setBurnAttribute("Id")
controls.setVectorDatatype(np.uint16)
infiles.points = pointfile
infiles.ts_image = os.path.join(dstDir, r'')
otherargs.pixels = np.zeros((30, 8), dtype=np.float32)
applier.apply(getPixelValues, infiles, outfiles, otherArgs=otherargs, controls=controls)
pixelValues = otherargs.pixels

# Graph time series statistics for each landform to assess seperability




################################################################################
# Train and test a random forest landform classification
################################################################################

def calc_accuracy(measured, modelled, uniqueClasses):
    """
    Calculates an error matrix by comparing measured and modelled classes, then
    calculates overall_accuracy, producers accuracy and users accuracy for all
    the classes.
    """
    errorMatrix = np.zeros((uniqueClasses.size, uniqueClasses.size), dtype=np.uint16)
    for row, cla in enumerate(uniqueClasses):
        for col, ref in enumerate(uniqueClasses):
            subCla = modelled[modelled == cla]
            subRef = measured[modelled == cla]
            errorMatrix[row, col] = np.sum(((subCla == cla) & (subRef == ref)))
    
    overall = 0.0
    users = np.zeros((uniqueClasses.size), dtype=np.float32)
    producers = np.zeros((uniqueClasses.size), dtype=np.float32)
    for row in range(uniqueClasses.size):
        for col in range(uniqueClasses.size):
            if row == col:
                overall += errorMatrix[row, col]
                if np.sum(errorMatrix[row, :]) > 0:
                    users[row] = errorMatrix[row, col] / np.sum(errorMatrix[row, :])
                else:
                    users[row] = 0
                if np.sum(errorMatrix[:, col]) > 0:
                    producers[row] = errorMatrix[row, col] / np.sum(errorMatrix[:, col])
                else:
                    producers[row] = 0
    overall = overall / float(np.sum(errorMatrix))
    
    accuracy = np.zeros((uniqueClasses.size * 2 + 1), dtype=np.float32)
    accuracy[0] = overall
    accuracy[1:uniqueClasses.size+1] = producers
    accuracy[uniqueClasses.size+1:] = users
    return accuracy


def train_rf_models(ids, variables, classes, n):
    """
    Trains the random forest models and saves the model files, iterating n times
    """
    uniqueClasses = np.unique(classes)
    # Create array for accuracy statistics for each iteration, which are
    # overall_accuracy, producers accuracy for the six classes and users
    # accuracy for the six classes
    accuracyStats = np.zeros((n, 2*uniqueClasses.size + 1), dtype=np.float32)
    
    for i in range(n):
    
        # Do a random split into training and testing samples (66/34)
        # If you want to startify by class then you need to change this
        sample = np.ones(ids.shape[0], dtype=np.uint8)
        sample[:int(0.66*ids.shape[0])] = 0
        np.random.shuffle(sample)
        training = (sample == 0)
        testing = (sample == 1)

        # Run the model
        rfc = RandomForestClassifier()
        rfc.fit(variables[training, :], classes[training])
        
        # Test the model
        predictions = rfc.predict(variables[testing, :])

        # Calculate accuracy statistics and add to array
        accuracyStats[i, :] = calc_accuracy(classes[testing], predictions, uniqueClasses)
        
        # Save the model file
        modelDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\dingo_fence\classify_landforms\rfmodels'
        if os.path.exists(modelDir) is False:
            os.mkdir(modelDir)
        joblib.dump(rfc, os.path.join(modelDir, 'rfmodel_%03d.pkl'%(i+1)))
        
    # Calculate accuracy statistics from array and output to csv file
    accuracyStats = 100 * accuracyStats
    with open(os.path.join(modelDir, 'accuracy.csv'), 'w') as f:
        f.write("Median overall accurcay (95% confidence limits) ")
        f.write("%.2f (%.2f-%.2f)\n"%(np.percentile(accuracyStats[:, 0], 50),
                                      np.percentile(accuracyStats[:, 0], 2.5),
                                      np.percentile(accuracyStats[:, 0], 97.5)))
        for i, c in enumerate(uniqueClasses):
            f.write("Class %s - median procucers accurcay (95%% confidence limits) "%c)
            f.write("%.2f (%.2f-%.2f)\n"%(np.percentile(accuracyStats[:, i+1], 50),
                                          np.percentile(accuracyStats[:, i+1], 2.5),
                                          np.percentile(accuracyStats[:, i+1], 97.5)))
            f.write("Class %s - median users accurcay (95%% confidence limits) "%c)
            f.write("%.2f (%.2f-%.2f)\n"%(np.percentile(accuracyStats[:, uniqueClasses.size+i+1], 50),
                                          np.percentile(accuracyStats[:, uniqueClasses.size+i+1], 2.5),
                                          np.percentile(accuracyStats[:, uniqueClasses.size+i+1], 97.5)))


# Get training data in integer format
classes = np.zeros_like(n)
classes[l == 'claypan'] = 1
classes[l == 'dune'] = 2
classes[l == 'swale'] = 3

# Train the random forest model
train_rf_models(n, stats, classes, 100)


################################################################################
# Apply the classifer
################################################################################

def model_classes(info, inputs, outputs):
    """
    Called through RIOS to create camphor classification and probability image.
    """
    # Sort out nodata values of the input images
    nodataValue = info.getNoDataValueFor(inputs.stats, band=1)
    nullArray = (inputs.stats[0] == nodataValue)
    
    # Reshape the input arrays to suit the models
    inshape = inputs.stats.shape
    variables = np.reshape(inputs.stats, (inshape[0],-1)).transpose().astype(np.float32)
    
    # Get the model files
    modelFiles = []
    modelDir = r'C:\Users\Adrian\OneDrive - UNSW\Documents\dingo_fence\classify_landforms\rfmodels'
    for modelFile in glob.glob(os.path.join(modelDir, '*.pkl')):
        modelFiles.append(modelFile)
    
    # Iterate through the models and make the predictions
    numModels = len(modelFiles)
    classArray = np.zeros((numModels, inshape[1], inshape[2]), dtype=np.uint8)
    for i, modelFile in enumerate(modelFiles):
        rfc = joblib.load(modelFile)
        prediction = rfc.predict(variables).astype(np.uint8)
        classArray[i, :, :] = np.reshape(prediction, (inshape[1], inshape[2]))
        classArray[i, :, :][nullArray == 1] = 0
    
    # Calculate the probability of each pixel being each class
    uniqueClasses = np.unique(classArray)
    probByClass = np.zeros((uniqueClasses.size, inshape[1], inshape[2]), dtype=np.float32)
    for i, c in enumerate(uniqueClasses):
        probByClass[i, :, :] = 100 * np.sum((classArray == c), axis=0) / float(numModels)
    probability = np.max(probByClass, axis=0).astype(np.uint8)
    
    # Determine the modal class
    classes = np.zeros((inshape[1], inshape[2]), dtype=np.uint8)
    for i, c in enumerate(uniqueClasses):
        classes[probByClass[i, :, :] == probability] = c
    
    # Create outputs
    outputs.classes = np.array([classes])
    outputs.probability = np.array([probability])


statsImage = r'E:\dingo_fence_landsat_fc\timeseries_stats.img'
infiles = applier.FilenameAssociations()
infiles.stats = statsImage
outfiles = applier.FilenameAssociations()
outfiles.classes = statsImage.replace('_stats.img', '_classes.img')
outfiles.probability =  statsImage.replace('_stats.img', '_probability.img')
controls = applier.ApplierControls()
controls.setStatsIgnore(0)
controls.setWindowXsize(20)
controls.setWindowYsize(20)
applier.apply(model_classes, infiles, outfiles, controls=controls)

clrTbl = np.array([[1, 204, 204, 204, 255],
                   [2, 255, 167, 127, 255],
                   [3, 255, 255, 190, 255]])
rat.setColorTable(outfiles.classes, clrTbl)