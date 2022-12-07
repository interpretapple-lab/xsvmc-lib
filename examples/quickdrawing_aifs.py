#!/usr/bin/env python
#
# This example shows how to use xSVMC to build augmented intuitionistic fuzzy sets.
# 
# The example is a modified version of the OpenCV tutorial entitled 'OCR of Hand-written Data using SVM' 
# posted on 
#       https://docs.opencv.org/4.5.3/dd/d3b/tutorial_py_svm_opencv.html
#
# Keywords: xSVMC, multiclass classification, contextualized classification
#

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from numpy.random.mtrand import randint
from sklearn import preprocessing
from common import hog

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xsvmlib.xsvmc import xSVMC

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

output_dir = 'output'  
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def get_article(word):
    art  = 'a'
    if word in ['ambulance', 'alarmclock']:
        art = 'an'
    elif word in ['sun']:
        art = 'the'
    elif word in ['binoculars']:
        art = ''
    else:
        art = 'a'
    return art


def run_example(output_dir, get_article):
    # Params
    BINS_N = 16 

    categories = ['ambulance', 'alarmclock', 'binoculars', 'bulldozer', 'crocodile',
              'hamburger', 'broom', 'castle', 'knee', 'snake',  'submarine', 'sun', 
              'telephone' , 'trumpet' ,'violin', 'guitar' ]

    n_objectsPerRow = 100
    n_rowsPerCategory = 10
    training_ratio = 0.7

    n_cats = len(categories)
    n_rows = n_cats * n_rowsPerCategory
    n_cols = n_objectsPerRow
    n_train_cols = int(n_objectsPerRow*training_ratio)
    n_test_cols = n_cols - n_train_cols 

    drawings = cv.imread(cv.samples.findFile('examples/quickdrawings.png'),0)
    if drawings is None:
        raise Exception("we need the quickdrawings.png image from samples/data here !")

    cells = [np.hsplit(row,n_cols) for row in np.vsplit(drawings,n_rows)]

# First part is trainData, remaining is testData
    train_cells = [ i[:n_train_cols] for i in cells ]
    test_cells = [ i[n_train_cols:] for i in cells]

    bin_n_data = [BINS_N for _ in train_cells]
    hog_data = [list(map(hog,row,bin_n_data)) for row in train_cells]
    X_train = np.float32(hog_data).reshape(-1,BINS_N*4)
    X_train = preprocessing.normalize(X_train, norm='l2')
    y_train = np.repeat(categories,n_train_cols*n_rowsPerCategory)[:,np.newaxis]

    bin_n_data = [BINS_N for _ in test_cells]
    hog_data = [list(map(hog,row,bin_n_data)) for row in test_cells]
    X_test = np.float32(hog_data).reshape(-1,BINS_N*4)
    X_test = preprocessing.normalize(X_test, norm='l2')
    y_test =  np.repeat(categories,n_test_cols*n_rowsPerCategory)[:,np.newaxis]

    kernel = 'poly'
    degree = 2.0
    gamma = 10.0
    coef0 = 0.0
    c0 =  0.1
    k = 2


    clf = xSVMC(kernel=kernel, C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
# Learning process
    print("Learning...")
    clf.fit(X_train, y_train.ravel())

    X_test2 = X_test[0:20]
    idx_category = 0
    print("Building AIFS '{0}'...".format(categories[idx_category]))
    aifs = clf.is_member_of(X_test2, idx_category)
    print("AIFS '{0}'".format(categories[idx_category]))
    for elem in aifs:
        print (" mu_hat: ({1:0.3f}, {2:3d}), nu_hat: ({3:0.3f}, {4:3d}), hesitation: {0:6.3f}".format( elem.hesitation, elem.mu_hat.value, elem.mu_hat.misv_idx, elem.nu_hat.value, elem.nu_hat.misv_idx))

   

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example(output_dir, get_article)
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))