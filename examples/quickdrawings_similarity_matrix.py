#!/usr/bin/env python
#
# This example shows how to use xSVMC to
#  - evaluate the membership (and nonmembership) of quick drawings in each of the classes;
#  - build augmented intuitionistic fuzzy sets (AIFSs)[1] from those evaluations;  
#  - compute the similarity levels among the resulting AIFSs; and
#  - build a similarity matrix between models of different classes of quick drawings.
# 
# Keywords: xSVMC, multiclass classification, contextualized classification, AIFS
#
# References:
# [1] M. Loor, G and De Tré, On the need for augmented appraisal degrees to handle experience-based evaluations,
#     Applied Soft Computing, Volume 54, 2017, Pages 284-295, ISSN 1568-4946,
#     https://doi.org/10.1016/j.asoc.2017.01.009. 

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from numpy.random.mtrand import randint
from sklearn import preprocessing
from common import hog
from similarity_SK2 import similarity


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

#  A few objects can be used for building an AIFS
    X_test2 = X_test[0:100]
    print("Evaluating memberships...")
    all_memberships_evals = clf.evaluate_all_memberships(X_test2) 

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
    if max_val_dif_cat * 1.1 < 1.0:
        max_val_dif_cat = max_val_dif_cat * 1.1

    import matplotlib.pyplot as plt
  
    fig, ax = plt.subplots()
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
    plt.savefig("{0}/quickdrawings_similarity_matrix.pdf".format(output_dir), dpi=300)
    plt.show()
   

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example(output_dir, get_article)
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))