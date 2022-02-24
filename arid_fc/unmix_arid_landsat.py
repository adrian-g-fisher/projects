#!/usr/bin/env python
"""
This creates arid zone fractional cover images from JRSRP surface reflectance
Landsat imagery. It follows the method as described in Shumack et al (2021).
Most of the code was developed by Sam Shumack, with minor modification by Adrian
Fisher. It needs AZN_M.csv, which contains the endmember data produced by Sam.

Shumack, S., Fisher, A., Hesse, P.P., (2021), Refining medium resolution
    fractional cover for arid Australia to detect vegetation dynamics and wind
    erosion susceptibility on longitudinal dunes. Remote Sensing of Environment,
    265, https://doi.org/10.1016/j.rse.2021.112647

This needs the fast-nnls package to be installed from GitHub with the following:
    pip install git+git://github.com/lostsea/fast-nnls.git

"""


import os
import sys
import argparse
import itertools
import numpy as np
from numpy.linalg import pinv
from sklearn.utils.extmath import randomized_svd
from fastnnls import fnnls
from rios import applier
from csv import reader


def fnnls(AtA, Aty, epsilon=None, iter_max=None):
    """
    Given a matrix A and vector y, find x which minimizes the objective function
    f(x) = ||Ax - y||^2.
    This algorithm is similar to the widespread Lawson-Hanson method, but
    implements the optimizations described in the paper
    "A Fast Non-Negativity-Constrained Least Squares Algorithm" by
    Rasmus Bro and Sumen De Jong.

    Note that the inputs are not A and y, but are
    A^T * A and A^T * y

    This is to avoid incurring the overhead of computing these products
    many times in cases where we need to call this routine many times.

    :param AtA:       A^T * A. See above for definitions. If A is an (m x n)
                      matrix, this should be an (n x n) matrix.
    :type AtA:        numpy.ndarray
    :param Aty:       A^T * y. See above for definitions. If A is an (m x n)
                      matrix and y is an m dimensional vector, this should be an
                      n dimensional vector.
    :type Aty:        numpy.ndarray
    :param epsilon:   Anything less than this value is consider 0 in the code.
                      Use this to prevent issues with floating point precision.
                      Defaults to the machine precision for doubles.
    :type epsilon:    float
    :param iter_max:  Maximum number of inner loop iterations. Defaults to
                      30 * [number of cols in A] (the same value that is used
                      in the publication this algorithm comes from).
    :type iter_max:   int, optional
    """
    if epsilon is None:
        epsilon = np.finfo(np.float64).eps

    n = AtA.shape[0]

    if iter_max is None:
        iter_max = 30 * n

    if Aty.ndim != 1 or Aty.shape[0] != n:
        raise ValueError('Invalid dimension; got Aty vector of size {}, ' \
                         'expected {}'.format(Aty.shape, n))

    # Represents passive and active sets.
    # If sets[j] is 0, then index j is in the active set (R in literature).
    # Else, it is in the passive set (P).
    sets = np.zeros(n, dtype=np.bool)
    # The set of all possible indices. Construct P, R by using `sets` as a mask
    ind = np.arange(n, dtype=int)
    P = ind[sets]
    R = ind[~sets]

    x = np.zeros(n, dtype=np.float64)
    w = Aty
    s = np.zeros(n, dtype=np.float64)

    i = 0
    # While R not empty and max_(n \in R) w_n > epsilon
    while not np.all(sets) and np.max(w[R]) > epsilon and i < iter_max:
        # Find index of maximum element of w which is in active set.
        j = np.argmax(w[R])
        # We have the index in MASKED w.
        # The real index is stored in the j-th position of R.
        m = R[j]

        # Move index from active set to passive set.
        sets[m] = True
        P = ind[sets]
        R = ind[~sets]

        # Get the rows, cols in AtA corresponding to P
        AtA_in_p = AtA[P][:, P]
        # Do the same for Aty
        Aty_in_p = Aty[P]

        # Update s. Solve (AtA)^p * s^p = (Aty)^p
        s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
        s[R] = 0.

        while np.any(s[P] <= epsilon):
            i += 1

            mask = (s[P] <= epsilon)
            alpha = np.min(x[P][mask] / (x[P][mask] - s[P][mask]))
            x += alpha * (s - x)

            # Move all indices j in P such that x[j] = 0 to R
            # First get all indices where x == 0 in the MASKED x
            zero_mask = (x[P] < epsilon)
            # These correspond to indices in P
            zeros = P[zero_mask]
            # Finally, update the passive/active sets.
            sets[zeros] = False
            P = ind[sets]
            R = ind[~sets]

            # Get the rows, cols in AtA corresponding to P
            AtA_in_p = AtA[P][:, P]
            # Do the same for Aty
            Aty_in_p = Aty[P]

            # Update s. Solve (AtA)^p * s^p = (Aty)^p
            s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
            s[R] = 0.

        x = s.copy()
        w = Aty - AtA.dot(x)

    return x


def add_interactive_terms(X):
    """
    This adds the interactive terms as used by Scarth et al. (2010), Guerschman
    et al. (2015), and Shumack et al (2021).
    """
    combos = list(itertools.combinations(range(0, X.shape[1]), 2))
    
    # bands 1-6 = reflectance
    
    # bands 7-12 = log transforms
    for b in range(0, 6):
        logb = np.log(X[:, b])
        X = np.hstack((X, logb.reshape((logb.shape[0], 1))))
    
    # bands 13-18 = product of log and original band
    for b in range(0, 6):
        logb = np.log(X[:, b]) * X[:, b]
        X = np.hstack((X,logb.reshape((logb.shape[0], 1))))
    
    # bands 19-33 = product of each band combination
    for c in combos:
        bprod = X[:, c[0]] * X[:, c[1]]
        X = np.hstack((X, bprod.reshape((bprod.shape[0], 1))))
    
    # bands 34-48 = product of each log band combination
    for c in combos:
        logbprod = X[:, c[0] + 6] * X[:, c[1] + 6]
        X = np.hstack((X, logbprod.reshape((logbprod.shape[0], 1))))
    
    # bands 49-63 = normalised band ratios
    for c in combos:
        norm_ratio = (X[:, c[0]] - X[:, c[1]]) / (X[:, c[0]] + X[:, c[1]])
        X = np.hstack((X, norm_ratio.reshape((norm_ratio.shape[0], 1))))
    
    return X



def unmix(X, M, w=1, c=3):
    """
    Unmixes an image into fractional cover.
    X is an array with n pixels times b bands (including all interactive terms).
    M is an array with c cover types times b bands.
    c = 3 for bare, PV and NPV
    w = 1 is the weighting factor for the sum-to-one constraint.
    """
    
    # Add ones for sum to one constraint
    Mw = np.hstack((M, (np.ones((M.shape[0], 1))) * w))
    X = np.hstack((X, (np.ones((X.shape[0], 1))) * w))
    
    # Create array to store fractional cover percentage values
    y_pred = np.zeros((X.shape[0], c), dtype=np.float32)
    
    # Iterate through the pixels
    for i, x in enumerate(X):
        solution = fnnls(Mw @ Mw.T, Mw @ x)
        y_pred[i] = solution
    
    return y_pred


def unmixImage(info, inputs, outputs, otherargs):
    """
    Called from rios to do the unmixing.
    """
    # Select only the blue, green, red, nir, swir1 and swir2 bands
    sr = inputs.sr[1:7]
    inshape = sr.shape
    
    # Remove zeros
    nodata = (inputs.sr[0] == 32767)
    sr[sr <= 0] = 1
    sr[sr == 32767] = 1
    
    # Reshape to n x b
    sr = np.reshape(sr, (inshape[0], inshape[1]*inshape[2])).transpose()

    # Rescale to floating point from 0 to 1
    sr = sr.astype(np.float32) / 10000.0
    
    # Add interactive terms
    sr = add_interactive_terms(sr)
    
    # Unmix and convert to percentages
    fc = unmix(sr, otherargs.M) * 100
    
    # Reshape back to an array with 3 bands
    fc = np.reshape(fc.transpose(), (3, inshape[1], inshape[2]))
    fc[:, nodata] = 255
    
    outputs.fc = fc.astype(np.uint8)


def main(inImage, outImage):
    """
    This sets up RIOS to do the unmixing.
    """
    # Retrieve endmembers from csv file
    endmembers = 'arid_fc_endmembers.csv'
    M = np.loadtxt(endmembers, delimiter=',')
    
    # Set up RIOS
    infiles = applier.FilenameAssociations()
    infiles.sr = inImage
    outfiles = applier.FilenameAssociations()
    outfiles.fc = outImage
    otherargs = applier.OtherInputs()
    otherargs.M = M
    controls = applier.ApplierControls()
    controls.setStatsIgnore(255)
    controls.setOutputDriverName("GTiff")
    applier.apply(unmixImage, infiles, outfiles, otherArgs=otherargs,
                  controls=controls)
    
    
def getCmdargs():
    """
    Get the command line arguments.
    """
    p = argparse.ArgumentParser(
            description=("Creates arid zone fractional cover images from " +
                         "JRSRP surface reflectance Landsat imagery"))
    p.add_argument("-i", "--inImage", dest="inImage", default=None,
                   help=("Input Landsat surface reflectance image file."))
    p.add_argument("-o", "--outImage", dest="outImage", default=None,
                   help=("Output arid fractional cover image file."))
    cmdargs = p.parse_args()
    if cmdargs.inImage is None or cmdargs.outImage is None:
        p.print_help()
        print("Must name inImage and outImage.")
        sys.exit()
    return cmdargs


if __name__ == "__main__":
    cmdargs = getCmdargs()
    main(cmdargs.inImage, outImage)