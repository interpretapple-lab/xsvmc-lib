#!/usr/bin/env python
#
# This example shows how to use xSVMC to
#  - evaluate the membership (and nonmembership) of handwritten numbers in each of the classes;
#  - build augmented intuitionistic fuzzy sets (AIFSs)[1] from those evaluations;  
#  - compute the similarity levels among the resulting AIFSs; and
#  - build a similarity matrix between models of different classes of handwritten numbers .
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
    y_test =  np.repeat(np.arange(10),250)[:,np.newaxis]

    degree = 2.0
    gamma = 5.383
    coef0 = 0.0
    c0 =  2.67
    k = 3

    clf = xSVMC(kernel='poly', C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
    print("Learning...")
    clf.fit(X_train, y_train.ravel())

    X_test2 = X_test[0:20]
    print("Evaluating memberships...")
    all_memberships_evals = [clf.evaluate_all_memberships(o) for o in X_test]

    n_cats = len(categories)
    sims = np.zeros((n_cats, n_cats))

    print("Computing similarities...")
    max_val_dif_cat = 0.0
    for idx_category in range(n_cats):
        for idx_category2 in range(idx_category + 1, n_cats):
            aifs = all_memberships_evals[idx_category]
            aifs2 = all_memberships_evals[idx_category2]
            sim_val = similarity(aifs, aifs2)
            sims[idx_category][idx_category2] = sim_val
            sims[idx_category2][idx_category] = sim_val
            if sim_val > max_val_dif_cat:
                max_val_dif_cat = sim_val
        sims[idx_category][idx_category] = 1.0 
   
    print("Visualizing results...")
    # For a better visualization 
    if max_val_dif_cat * 1.2 < 1.0:
        max_val_dif_cat = max_val_dif_cat * 1.2

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(sims, cmap="Greys_r", vmax= max_val_dif_cat )
   

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))

    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('similarity level', rotation=90, va="top")

    plt.setp(ax.get_xticklabels(), rotation=90, va="center", ha="left", rotation_mode="anchor")

    fig.tight_layout()
    plt.savefig("{0}/digits_similarity_matrix.pdf".format(output_dir), dpi=300)
    plt.show()
   
   

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example()
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))