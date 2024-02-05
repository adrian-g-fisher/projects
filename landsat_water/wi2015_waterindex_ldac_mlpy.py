#!/usr/bin/env python

"""

 This reads in the training data extracted for all pixels and centre pixels,
 creates indexes using ldac in mlpy, conducts a 10-fold cross validation of
 classification accuracy, and writes all output to a text file. There are 48
 different models tested:
    - dbf (no topo correction) and dbg (with topo correction) data (dbf/dbg)
    - all pixels and centre pixels (all/centre)
    - 2-class and 3-class LDAC (2c/3c)
    - no tranforms, log transforms, quadratic transforms (raw/log/quad)
    - using all bands and all but the blue band (allbands/noblue)
 
"""

import numpy as np
import mlpy
import sys


def get_accuracy(y, z):
    """
    Makes an error matrix from reference (y) and prediction (z) vectors.
    """
    a = np.sum(np.where(y == -1, 0, np.where(z == -1, 0, 1)))
    b = np.sum(np.where(y ==  1, 0, np.where(z == -1, 0, 1)))
    c = np.sum(np.where(y == -1, 0, np.where(z ==  1, 0, 1)))
    d = np.sum(np.where(y ==  1, 0, np.where(z ==  1, 0, 1)))
    com = 100*b/float(a+b)
    omi = 100*c/float(a+c)
    acc = 100*(a+d)/float(a+b+c+d)
    fpr = 100*b/float(b+d)
    return (acc, com, omi, fpr)


def shuffle_array(inArray):
    """
    Subsets an array to the nearest 10, and randomly assigns rows to classes 1-10.
    """
    n = 1
    numPixels = np.shape(inArray)[0]
    subNum = 10 * (numPixels / 10)
    np.random.shuffle(inArray)
    subSample = inArray[0:subNum, :]
    n = np.shape(subSample)[0]
    classes = np.zeros(n, dtype=int)
    classes[0:(n/10)] = 1
    classes[(n/10):(2*n/10)] = 2
    classes[(2*n/10):(3*n/10)] = 3
    classes[(3*n/10):(4*n/10)] = 4
    classes[(4*n/10):(5*n/10)] = 5
    classes[(5*n/10):(6*n/10)] = 6
    classes[(6*n/10):(7*n/10)] = 7
    classes[(7*n/10):(8*n/10)] = 8
    classes[(8*n/10):(9*n/10)] = 9
    classes[(9*n/10):-1] = 10
    classes[classes == 0] = 10
    np.random.shuffle(classes)
    return (subSample, classes)


# Load data
accuracy_file = "wi2015_waterindex_accuracy.txt"
with open(accuracy_file, "w") as f:
    f.write("sample, overall accuracy, water commission, water omision, false positive rate\n")

equation_file = "wi2015_waterindex_equations.txt"
with open(equation_file, "w") as f:
    f.write("sample, equation number, constant, coefficients\n")

training_files = ["/mnt/project/landsat_water/jrsrp_method/new_training_values/wi2015_training_all.txt",
                  "/mnt/project/landsat_water/jrsrp_method/new_training_values/wi2015_training_centre.txt"]
p = ["all", "centre"]
t = ["dbf", "dbg"]
c = ["2c", "3c"]
b = ["allbands", "noblue"]
i = ["raw", "log", "quad"]

for l1, training in enumerate(training_files):
    P = p[l1]
    dbfData = []
    dbgData = []
    with open(training, "r") as f:
        f.readline()
        for line in f:
            # Format: Scene, Year, Class (water=1, nonwater=2, shadow=3),
            #         dbf_b1, dbf_b2, dbf_b3, dbf_b4, dbf_b5, dbf_b7,
            #         dbg_b1, dbg_b2, dbg_b3, dbg_b4, dbg_b5, dbg_b7
            (s, y, cl, dbf_b1, dbf_b2, dbf_b3, dbf_b4, dbf_b5, dbf_b7,
             dbg_b1, dbg_b2, dbg_b3, dbg_b4, dbg_b5, dbg_b7) = line.split(",")
            dbfData.append([int(cl), int(dbf_b1), int(dbf_b2), int(dbf_b3), int(dbf_b4), int(dbf_b5), int(dbf_b7)])
            dbgData.append([int(cl), int(dbg_b1), int(dbg_b2), int(dbg_b3), int(dbg_b4), int(dbg_b5), int(dbg_b7)])
    dbfData = np.asarray(dbfData)
    dbgData = np.asarray(dbgData)

    # Recode shadow (3) into non-water (2)
    dbfData[dbfData[:,0] == 3] = 2
    dbgData[dbfData[:,0] == 3] = 2

    for l2, trainingData in enumerate([dbfData, dbgData]):
        T = t[l2]
        
        # 2 or 3 classes
        for C in c:
            waterData = trainingData[trainingData[:,0] == 1]
            otherData = trainingData[trainingData[:,0] == 2]
        
            if C == "2c":
                
                # Shuffle and assign to 10 classes
                (waterData, waterClasses) = shuffle_array(waterData)
                (otherData, otherClasses) = shuffle_array(otherData)
                classes = np.concatenate((waterClasses, otherClasses), axis=0)
                
                # Create labels: water (1) and nonwater (-1)
                waterLabels = np.ones((np.shape(waterData)[0], 1), dtype=int)
                otherLabels = -1 * np.ones((np.shape(otherData)[0], 1), dtype=int)
            
                x = np.concatenate((waterData, otherData), axis=0)
                y = np.concatenate((waterLabels, otherLabels), axis=0)
                
            if C == "3c":
                # Sort into clear and coloured water
                # clear water has b1 - b2 > -0.011 and
                #                 b1 + b2 <  0.053
                diff = waterData[:,1]/10000.0 - waterData[:,2]/10000.0
                add = waterData[:,1]/10000.0 + waterData[:,2]/10000.0
                watertype = np.where(diff > -0.011, np.where(add < 0.053, 0, 1), 1)
                clearData = waterData[watertype == 0]
                colouredData = waterData[watertype == 1]
                
                # Shuffle and assign to 10 classes
                (clearData, clearClasses) = shuffle_array(clearData)
                (colouredData, colouredClasses) = shuffle_array(colouredData)
                (otherData, otherClasses) = shuffle_array(otherData)
                classes = np.concatenate((clearClasses, colouredClasses, otherClasses), axis=0)
                
                # Create labels: water (0), coloured water (1) and nonwater (2)
                clearLabels = np.zeros((np.shape(clearData)[0], 1), dtype=int)
                colouredLabels = np.ones((np.shape(colouredData)[0], 1), dtype=int)
                otherLabels = 2 * np.ones((np.shape(otherData)[0], 1), dtype=int)
                
                x = np.concatenate((clearData, colouredData, otherData), axis=0)
                y = np.concatenate((clearLabels, colouredLabels, otherLabels), axis=0)
                
            # Use blue or not
            for B in b:
                
                if B == 'allbands':
                    x1 = x[:,1:]
                    
                if B == 'noblue':
                    x1 = x[:,2:]

                # Transform inputs
                for I in (i):
                    
                    if I == "log":
                        x2 = np.log1p(x1)
                        
                    else:
                        x2 = np.copy(x1)
                    
                    if B == "allbands":
                        interaction = np.zeros((np.shape(x)[0], 15), dtype=float)
                        interaction[:, 0] = x2[:, 0] * x2[:, 1] # b1b2
                        interaction[:, 1] = x2[:, 0] * x2[:, 2] # b1b3
                        interaction[:, 2] = x2[:, 0] * x2[:, 3] # b1b4
                        interaction[:, 3] = x2[:, 0] * x2[:, 4] # b1b5
                        interaction[:, 4] = x2[:, 0] * x2[:, 5] # b1b7
                        interaction[:, 5] = x2[:, 1] * x2[:, 2] # b2b3
                        interaction[:, 6] = x2[:, 1] * x2[:, 3] # b2b4
                        interaction[:, 7] = x2[:, 1] * x2[:, 4] # b2b5
                        interaction[:, 8] = x2[:, 1] * x2[:, 5] # b2b7
                        interaction[:, 9] = x2[:, 2] * x2[:, 3] # b3b4
                        interaction[:,10] = x2[:, 2] * x2[:, 4] # b3b5
                        interaction[:,11] = x2[:, 2] * x2[:, 5] # b3b7
                        interaction[:,12] = x2[:, 3] * x2[:, 4] # b3b5
                        interaction[:,13] = x2[:, 3] * x2[:, 5] # b3b7
                        interaction[:,14] = x2[:, 4] * x2[:, 5] # b4b7
                    if B == "noblue":
                        interaction = np.zeros((np.shape(x)[0], 10), dtype=float)
                        interaction[:, 0] = x2[:, 0] * x2[:, 1] # b2b3
                        interaction[:, 1] = x2[:, 0] * x2[:, 2] # b2b4
                        interaction[:, 2] = x2[:, 0] * x2[:, 3] # b2b5
                        interaction[:, 3] = x2[:, 0] * x2[:, 4] # b2b7
                        interaction[:, 4] = x2[:, 1] * x2[:, 2] # b3b4
                        interaction[:, 5] = x2[:, 1] * x2[:, 3] # b3b5
                        interaction[:, 6] = x2[:, 1] * x2[:, 4] # b3b7
                        interaction[:, 7] = x2[:, 2] * x2[:, 3] # b4b5
                        interaction[:, 8] = x2[:, 2] * x2[:, 4] # b4b7
                        interaction[:, 9] = x2[:, 3] * x2[:, 4] # b5b7
                    
                    if I == "quad":
                        x3 = np.concatenate((x2, interaction, np.square(x2)), axis=1)
                        
                    else:
                        x3 = np.concatenate((x2, interaction), axis=1)
                    
                    # LDAC
                    ldac = mlpy.LDAC()
                    ldac.learn(x3, np.ndarray.flatten(y))
                    bias = ldac.bias()
                    w = ldac.w()

                    # Do 10-fold cross validation
                    ylist = []
                    zlist = []

                    for k in range(1, 11):
    
                        # Create discriminant function
                        xk = x3[classes != k]
                        yk = np.ndarray.flatten(y[classes != k])
                        ldack = mlpy.LDAC()
                        ldack.learn(xk, yk)
                        bk = ldack.bias()
                        wk = ldack.w()
    
                        # Test function
                        xj = x3[classes == k]
                        yj = np.ndarray.flatten(y[classes == k])
                        z = ldack.pred(xj)
    
                        # Add result to list
                        ylist.append(yj)
                        zlist.append(z)

                    # Create error matrix for cross-validation
                    ytotal = np.vstack(ylist)
                    ztotal = np.vstack(zlist)
            
                    # Collapse 3 classes into 2 before calculating errors
                    if np.min(ytotal) == 0:
                        ytotal[ytotal == 0] = 1
                        ytotal[ytotal == 2] = -1
                        ztotal[ztotal == 0] = 1
                        ztotal[ztotal == 2] = -1
            
                    # Calculate errors
                    (acc, com, omi, fpr) = get_accuracy(ytotal, ztotal)

                    # Output accuracy to textfile
                    sample = "%s_%s_%s_%s_%s"%(P, C, T, B, I)
                    accuracy = "%.2f, %.2f, %.2f, %.2f"%(acc, com, omi, fpr)
                    with open(accuracy_file, "a") as f:
                        f.write("%s, %s\n"%(sample, accuracy))
                    
                    # Output equation to textfile
                    if np.shape(bias) == (3,):
                        for R in [0, 1, 2]:
                            coefficients = "%.4f"%bias[R]
                            for Coef in w[R]:
                                coefficients = "%s, %.4f"%(coefficients, Coef)
                            with open(equation_file, "a") as f:
                                f.write("%s, %s, %s\n"%(sample, R+1, coefficients))
                            
                    else:
                        coefficients = "%.4f"%bias
                        for Coef in w:
                            coefficients = "%s, %.4f"%(coefficients, Coef)
                        with open(equation_file, "a") as f:
                            f.write("%s, 1, %s\n"%(sample, coefficients))
            
                    print sample
            
