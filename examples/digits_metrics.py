#!/usr/bin/env python
#
# This example shows how to evaluate the performance of xSVMC while performing contextualized predictions of the
# classes of handwritten numbers. 
# 
# Keywords: xSVMC, multiclass classification, contextualized classification
#

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from common import hog, ordinal, ordinalltx
import os, sys
import time

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


def run_metrics():
    # Params
    BINS_N = 16 

    categories = ['zero', 'one', 'two', 'three', 'four',
              'five', 'six', 'seven', 'eight', 'nine' ]

    n_objectsPerRow = 100
    n_rowsPerCategory = 5
    training_ratio = 0.5

    n_cats = len(categories)
    n_rows = n_cats * n_rowsPerCategory
    n_cols = n_objectsPerRow
    n_train_cols = int(n_objectsPerRow*training_ratio)
    n_test_cols = n_cols - n_train_cols 

    digits = cv.imread(cv.samples.findFile('examples/digits.png'),0)
    if digits is None:
        raise Exception("we need the digits.png image from samples/data here !")

    cells = [np.hsplit(row,n_cols) for row in np.vsplit(digits,n_rows)]

    # First part is trainData, remaining is testData
    train_cells = [ i[:n_train_cols] for i in cells ]
    test_cells = [ i[n_train_cols:] for i in cells]

    hog_data = [list(map(hog,row)) for row in train_cells]
    X_train = np.float32(hog_data).reshape(-1,BINS_N*4)
    X_train = preprocessing.normalize(X_train, norm='l2')
    y_train = np.repeat(categories,n_train_cols*n_rowsPerCategory)[:,np.newaxis]

    hog_data = [list(map(hog,row)) for row in test_cells]
    X_test = np.float32(hog_data).reshape(-1,BINS_N*4)
    X_test = preprocessing.normalize(X_test, norm='l2')
    y_test =  np.repeat(categories,n_test_cols*n_rowsPerCategory)[:,np.newaxis]

    kernel = 'poly'
    degree = 2.0
    gamma = 5.383
    coef0 = 0.0
    c0 =  2.67
    k = 2


    clf = xSVMC(kernel=kernel, C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
    # Learning process
    print("Learning...")
    clf.fit(X_train, y_train.ravel())

    predictions = clf.predict(X_test) 
    from sklearn.metrics import accuracy_score
    print("Accuracy (SVC-predictions): {0}".format( accuracy_score(y_test, predictions)))
    

    # ---- Evaluation process: Begin --- 
    from multiprocessing import Pool as ThreadPool

    print("Evaluating {0} objects...".format(len(y_test)))
    NUMBER_OF_PROCESSES = 16
    
    pool = ThreadPool(NUMBER_OF_PROCESSES)
    ctx_predictions = pool.map(clf.predict_with_context_by_voting, X_test)
    pool.close()
    pool.join()

     # ---- Evaluation process: End --- 

    class_cntrs = {}

    if len(ctx_predictions) == len(y_test):
        for idx_obj in range(len(y_test)):
            prediction = ctx_predictions[idx_obj]
            actual_class =  y_test[idx_obj][0]

            if actual_class not in class_cntrs:
                class_cntrs[actual_class] = np.zeros(k + 1)

            inTopK = False
            for rank in range(k):
                if prediction[rank].class_name == actual_class:
                    class_cntrs[actual_class][rank] = class_cntrs[actual_class][rank] + 1
                    inTopK = True
                    break
            if not inTopK:
                class_cntrs[actual_class][k] = class_cntrs[actual_class][k] + 1

    n_wrong_topK = 0
    n_right_top1 = 0
    for key in class_cntrs:
        cntr = class_cntrs[key]
        print('\n{0}'.format(key))

        support = cntr[k]
        n_wrong_topK += cntr[k]
        n_right_top1 += cntr[0]
        for rank in range(k):
           support += cntr[rank]
       
        for rank in range(k):
            print("\t{0}: \t\t{1:5d} ({2:3.2f}%)".format(ordinal(rank+1), int(cntr[rank]), 100*cntr[rank]/support))
        print("\tNot in top {2}: \t{0:5d} ({1:3.2f}%)".format(int(cntr[k]), 100*cntr[k]/support, k))

    n = len(y_test) 
    n_right_topK = n - n_wrong_topK
    n_wrong_top1 = n - n_right_top1

    print('\nOverall results:')

    print("\tIn top k: \t{0:5d} ({1:3.2f}%)".format(int(n_right_topK), 100*n_right_topK/n))
    print("\tNot in top k: \t{0:5d} ({1:3.2f}%)\n".format(int(n_wrong_topK), 100*n_wrong_topK/n))

    print("\tIn top 1: \t{0:5d} ({1:3.2f}%)".format(int(n_right_top1), 100*n_right_top1/n))
    print("\tNot in top 1: \t{0:5d} ({1:3.2f}%)\n".format(int(n_wrong_top1), 100*n_wrong_top1/n))

    print("Accuracy (xSVC-predictions): {0:.5f}".format(n_right_topK/n))

if __name__ == '__main__':

    start_time = time.time()
    run_metrics()
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))