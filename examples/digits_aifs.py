#!/usr/bin/env python
#
# This example shows how to use xSVMC to perform contextualized predictions in 
# multiclass classification. It also shows how to evaluate the membership (and
# nonmembership) of handwritten numbers in each of the classes.
# 
# The example is a modified version of the digits example (digits.py).
#
# Keywords: xSVMC, multiclass classification, contextualized classification
#

import cv2 as cv
import numpy as np
from sklearn import preprocessing
import os, sys
from common import deskew, hog

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xsvmlib.xsvmc import xSVMC

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_


def run_example():
    # Params
    RASTER_SZ = 20
    BINS_N = 16

    digits = cv.imread(cv.samples.findFile('examples/digits.png'),0)
    if digits is None:
        raise Exception("we need the digits.png image from samples/data here !")

    cells = [np.hsplit(row,100) for row in np.vsplit(digits,50)]

    categories = ['zero', 'one', 'two', 'three', 'four',
              'five', 'six', 'seven', 'eight', 'nine' ]
# First half is trainData, remaining is testData
    train_cells = [ i[:50] for i in cells ]
    test_cells = [ i[50:] for i in cells]

    digits_coord = [[i,j] for i in range(50) for j in range(50)]

    raster_sz_data = [RASTER_SZ for _ in train_cells]
    deskewed = [list(map(deskew,row, raster_sz_data)) for row in train_cells]
    bin_n_data = [BINS_N for _ in deskewed]
    hog_data = [list(map(hog,row, bin_n_data)) for row in deskewed]

    X_train = np.float32(hog_data).reshape(-1,BINS_N*4)
    X_train = preprocessing.normalize(X_train, norm='l2')
    y_train = np.repeat(np.arange(10),250)[:,np.newaxis]

    raster_sz_data = [RASTER_SZ for _ in test_cells]
    deskewed = [list(map(deskew,row, raster_sz_data)) for row in test_cells]
    bin_n_data = [BINS_N for _ in deskewed]
    hog_data = [list(map(hog,row, bin_n_data)) for row in deskewed]

    X_test = np.float32(hog_data).reshape(-1,BINS_N*4)
    X_test = preprocessing.normalize(X_test, norm='l2')
    y_test =  np.repeat(np.arange(10),250)[:,np.newaxis]

    degree = 2.0
    gamma = 5.383
    coef0 = 0.0
    c0 =  2.67
    k = 3

    clf = xSVMC(kernel='poly', C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
    clf.fit(X_train, y_train.ravel())

    X_test2 = X_test[0:1000]
    idx_category = 0
    print("Building AIFS '{0}'...".format(categories[idx_category]))

    # aifs = [clf.is_member_of(o, idx_category) for o in X_test2] # too slow...not recommended
    aifs = clf.is_member_of(X_test2, idx_category)  # recommended

    print("AIFS '{0}'".format(categories[idx_category]))
    for elem in aifs:
        print (" mu_hat: ({1:0.3f}, {2:3d}), nu_hat: ({3:0.3f}, {4:3d}), hesitation: {0:6.3f}".format( elem.hesitation, elem.mu_hat.value, elem.mu_hat.misv_idx, elem.nu_hat.value, elem.nu_hat.misv_idx))

 

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example()
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))