#!/usr/bin/env python
"""

"""

import os
import sys
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
from rios import applier, rat
from sklearn.ensemble import RandomForestClassifier

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans',
          'font.sans-serif': 'Arial', 'font.family': 'sans-serif'}
plt.rcParams.update(params)


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
            otherargs.pixels[ids[i]-1, 0] = ids[i]
            otherargs.pixels[ids[i]-1, 1:] = stats

def extract_training():
    """
    Gets the pixel values for points and saves to a CSV file.
    """
    pointfile = r'S:\witchelina\awp_distance\classification_trainingdata\Paradise_CrkStn_trainingdatapoints.shp'
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(pointfile, 0)
    layer = dataSource.GetLayer()
    pointIds = []
    pointClasses = []
    for feature in layer:
        pointIds.append(int(feature.GetField("Id")))
        pointClasses.append(feature.GetField("Class"))
    pointIds = np.array(pointIds)
    pointClasses = np.array(pointClasses)
    
    imagefile = r'S:\witchelina\timeseries_statistic_images\timeseries_stats_198712202302.tif'
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    controls.setBurnAttribute("Id")
    controls.setVectorDatatype(np.uint16)
    infiles.points = pointfile
    infiles.ts_image = imagefile
    otherargs.pixels = np.zeros((200, 13), dtype=np.float32)
    applier.apply(getPixelValues, infiles, outfiles, otherArgs=otherargs, controls=controls)
    pixelValues =  otherargs.pixels
    pixelValues = pixelValues[pixelValues[:, 0].argsort()]
    colnames = ['ID', 'Landform',
                'PV_mean', 'PV_stdev', 'PV_min', 'PV_max',
                'NPV_mean', 'NPV_stdev', 'NPV_min', 'NPV_max',
                'Bare_mean', 'Bare_stdev', 'Bare_min', 'Bare_max']

    with open('landform_training.csv', 'w') as f:
        f.write('%s\n'%','.join(colnames))
        for i in range(pointIds.size):
            line = '%i'%pixelValues[i, 0]
            line = '%s,%i'%(line, pointClasses[i])
            for x in range(1, 13):
                line = '%s,%.4f'%(line, pixelValues[i, x])
            f.write('%s\n'%line)


def graph_training():
    """
    Reads in the training data and makes boxplots showing how different
    statistics can separate the two classes.
    1 - Stony hills and plains
    2 - Watercourses and swamps
    """
    pixelvalues = np.genfromtxt('landform_training.csv', delimiter=',',
                                names=True)
    
    statistics = [['PV_mean', 'PV_stdev', 'PV_min', 'PV_max'],
                  ['NPV_mean', 'NPV_stdev', 'NPV_min', 'NPV_max'],
                  ['Bare_mean', 'Bare_stdev', 'Bare_min', 'Bare_max']]
                  
    fig, axs = plt.subplots(3, 4)
    fig.set_size_inches((8, 8))

    for i in range(3):
        for j in range(4):
            axs[i, j].set_title(statistics[i][j])
            
            hills = pixelvalues[statistics[i][j]][pixelvalues['Landform'] == 1]
            creek = pixelvalues[statistics[i][j]][pixelvalues['Landform'] == 2]
            bp = axs[i, j].boxplot([hills, creek], patch_artist=True)
            plt.setp(bp['boxes'], color='black')
            plt.setp(bp['whiskers'], color='black')
            plt.setp(bp['fliers'], color='black', marker='+')
            plt.setp(bp['medians'], color='black')
            colors = ['pink', 'lightblue']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                
            axs[i, j].set_xticks([])
            axs[i, j].set_xticklabels([])
    
    axs[2, 2].legend([bp["boxes"][0], bp["boxes"][1]],
                     ['Stony hills and plains', 'Watercourses and swamps'],
                     ncol = 2, frameon=False, bbox_to_anchor=(1.6, -0.01))
    
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(r'landform_boxplots.png', dpi=300)


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


def train_rf_models():
    """
    Trains the random forest models and saves the model files, iterating n times
    """
    pixelvalues = np.genfromtxt('landform_training.csv', delimiter=',',
                                names=True)
    ids = pixelvalues['ID'].astype(np.uint8)
    classes = pixelvalues['Landform'].astype(np.uint8)
    variables = np.transpose(np.vstack([pixelvalues['PV_mean'], pixelvalues['PV_stdev'],
                           pixelvalues['PV_min'], pixelvalues['PV_max'],
                           pixelvalues['NPV_mean'], pixelvalues['NPV_stdev'],
                           pixelvalues['NPV_min'], pixelvalues['NPV_max'],
                           pixelvalues['Bare_mean'], pixelvalues['Bare_stdev'],
                           pixelvalues['Bare_min'], pixelvalues['Bare_max']]))
    
    n = 100
    
    uniqueClasses = np.unique(classes)
    # Create array for accuracy statistics for each iteration, which are
    # overall_accuracy, producers accuracy for the two classes and users
    # accuracy for the two classes
    accuracyStats = np.zeros((n, 2*uniqueClasses.size + 1), dtype=np.float32)
    
    for i in range(n):
    
        # Do a random split into training and testing samples (66/34)
        # stratifying by class.
        sample_hills = np.ones(int(ids.shape[0]/2), dtype=np.uint8)
        sample_hills[:int(0.66*ids.shape[0]/2)] = 0
        np.random.shuffle(sample_hills)
        
        sample_creeks = np.ones(int(ids.shape[0]/2), dtype=np.uint8)
        sample_creeks[:int(0.66*ids.shape[0]/2)] = 0
        np.random.shuffle(sample_creeks)
        
        sample = np.hstack([sample_hills, sample_creeks])
        
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
        modelDir = r'S:\witchelina\awp_distance\classification_trainingdata\rfmodels'
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


def model_classes(info, inputs, outputs):
    """
    Called through RIOS to create classification and probability image.
    """
    # Sort out nodata values of the input images
    nodataValue = info.getNoDataValueFor(inputs.stats, band=1)
    nullArray = (inputs.stats[0] == nodataValue)
    
    # Reshape the input arrays to suit the models
    inshape = inputs.stats.shape
    variables = np.reshape(inputs.stats, (inshape[0],-1)).transpose().astype(np.float32)
    
    # Get the model files
    modelFiles = []
    modelDir = r'S:\witchelina\awp_distance\classification_trainingdata\rfmodels'
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


def apply_rf_models():
    """
    Applies the random forest model to the image data.
    """
    statsImage = r'S:\witchelina\timeseries_statistic_images\timeseries_stats_198712202302.tif'
    outdir = r'C:\Users\Adrian\Documents\witchelina'
    infiles = applier.FilenameAssociations()
    infiles.stats = statsImage
    outfiles = applier.FilenameAssociations()
    outfiles.classes = os.path.join(outdir, 'landform_classes.img')
    outfiles.probability =  os.path.join(outdir, 'landform_probability.img')
    controls = applier.ApplierControls()
    controls.setStatsIgnore(0)
    controls.setWindowXsize(20)
    controls.setWindowYsize(20)
    applier.apply(model_classes, infiles, outfiles, controls=controls)
    clrTbl = np.array([[1, 204, 204, 204, 255],
                       [2, 255, 167, 127, 255]])
    rat.setColorTable(outfiles.classes, clrTbl)


# Run the different functions
#extract_training()
#graph_training()
train_rf_models()
#apply_rf_models()