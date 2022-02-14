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

    idx_obj = 0

    for i in range(10):
        x = X_test[idx_obj]
        print("----- Object {0} -----".format(idx_obj))

        print("- Top-{0} Predictions".format(k))
        topK_predictions = clf.predict_with_context(x)
        topK_predictions_2 = clf.predict_with_context_by_voting(x)
        for j in range(k):
            print("  buoyancy-rank top-{1} class: {0}, misv+: {3:3d}, misv-: {4:3d}, buoyancy: {2:6.3f}".format(topK_predictions[j].class_name, j+1, topK_predictions[j].eval.buoyancy, topK_predictions[j].eval.mu_hat.misv_idx, topK_predictions[j].eval.nu_hat.misv_idx))
            print("     votes-rank top-{1} class: {0}, misv+: {3:3d}, misv-: {4:3d}, buoyancy: {2:6.3f}".format(topK_predictions_2[j].class_name, j+1, topK_predictions_2[j].eval.buoyancy, topK_predictions_2[j].eval.mu_hat.misv_idx, topK_predictions_2[j].eval.nu_hat.misv_idx))
   
        print("- Memberships ")
        evals = clf.evaluate_all_memberships(x)
        for idx_class in range(len(clf.classes_)):
            ret1 = evals[idx_class]
            print ("  idx_class: {0}, mu_hat: ({2:0.3f}, {3:3d}), nu_hat: ({4:0.3f}, {5:3d}), buoyancy: {1:6.3f}".format(idx_class, ret1.buoyancy, ret1.mu_hat.value, ret1.mu_hat.misv_idx, ret1.nu_hat.value, ret1.nu_hat.misv_idx))

        idx_obj += 250
        print("---")

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example()
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))