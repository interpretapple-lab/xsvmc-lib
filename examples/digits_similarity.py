#!/usr/bin/env python
#
# This example shows how to use xSVMC to
#  - evaluate the membership (and nonmembership) of handwritten numbers in each of the classes;
#  - build augmented intuitionistic fuzzy sets (AIFSs)[1] from those evaluations; and  
#  - compute the similarity levels among the AIFSs produced by three different classifiers.
# 
# Keywords: xSVMC, multiclass classification, contextualized classification, AIFS
#
# References:
# [1] M. Loor, G and De Tré, On the need for augmented appraisal degrees to handle experience-based evaluations,
#     Applied Soft Computing, Volume 54, 2017, Pages 284-295, ISSN 1568-4946,
#     https://doi.org/10.1016/j.asoc.2017.01.009. 


import cv2 as cv
import numpy as np
from sklearn import preprocessing
import os, sys
from common import deskew, hog
from similarity_SK2 import similarity


_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xsvmlib.xsvmc import xSVMC
from xsvmlib.xmodels import xAIFSElement

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

output_dir = 'output'  
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
    # y_test =  np.repeat(np.arange(10),250)[:,np.newaxis]

    # Parameters (Classifier A)
    degree_A = 2.0
    gamma_A = 5.383
    coef0_A = 0.0
    c0_A =  2.67

    # Parameters (Classifier B)
    degree_B = 2.0
    gamma_B = 3.383
    coef0_B = 0.0
    c0_B =  2.67

    # Parameters (Classifier C)
    degree_C = 3.0
    gamma_C = 5.383
    coef0_C = 0.0
    c0_C =  2.67
    
    k = 3

    clf_A = xSVMC(kernel='poly', C=c0_A, degree=degree_A, gamma=gamma_A, coef0=coef0_A, k = k)
    print("Learning (classifier A)...")
    clf_A.fit(X_train, y_train.ravel())

    clf_B = xSVMC(kernel='poly', C=c0_B, degree=degree_B, gamma=gamma_B, coef0=coef0_B, k = k)
    print("Learning (classifier B)...")
    clf_B.fit(X_train, y_train.ravel())

    clf_C = xSVMC(kernel='poly', C=c0_C, degree=degree_C, gamma=gamma_C, coef0=coef0_C, k = k)
    print("Learning (classifier C)...")
    clf_C.fit(X_train, y_train.ravel())

    X_test2 = X_test[0:20]
    print("Evaluating memberships (classifier A)...")
    all_memberships_evals_A = [clf_A.evaluate_all_memberships(o) for o in X_test2]

    print("Evaluating memberships (classifier B)...")
    all_memberships_evals_B = [clf_B.evaluate_all_memberships(o) for o in X_test2]

    print("Evaluating memberships (classifier C)...")
    all_memberships_evals_C = [clf_C.evaluate_all_memberships(o) for o in X_test2]

    n_cats = len(categories)
    sims_A_vs_B = np.zeros(n_cats)
    sims_A_vs_C = np.zeros(n_cats)

    print("Computing similarities...")

    for idx_category in range(n_cats):
        aifs_A = all_memberships_evals_A[idx_category]
        aifs_B = all_memberships_evals_B[idx_category]
        aifs_C = all_memberships_evals_C[idx_category]
        sim_val_A_vs_B = similarity(aifs_A, aifs_B)
        sim_val_A_vs_C = similarity(aifs_A, aifs_C)
        sims_A_vs_B[idx_category] = sim_val_A_vs_B
        sims_A_vs_C[idx_category] = sim_val_A_vs_C
           
    print("Visualizing results...")

    import matplotlib.pyplot as plt
  
    fig, ax = plt.subplots(figsize=(4,3))
    scatters = []
    scatters.append(ax.scatter(categories, sims_A_vs_B, marker= '+', c='black'))
    scatters.append(ax.scatter(categories, sims_A_vs_C, marker=',', c='gray'))
   

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    ax.legend(scatters, ["A vs. B","A vs. C"])

    plt.setp(ax.get_xticklabels(), rotation=90, va="center", ha="right", rotation_mode="anchor")

    fig.tight_layout()
    plt.savefig("{0}/digits_similarity.pdf".format(output_dir), dpi=300)
    plt.show()
   
   

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example()
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))