#!/usr/bin/env python
#
# This example shows how to use xSVMC to perform contextualized predictions of the
# classes of handwritten numbers. 
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
    if word in ['one', 'eight']:
        art = 'an'
    else:
        art = 'a'
    return art


def run_test(output_dir, get_article):
    bin_n = 16 # Number of bins
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

    train_coords = [[j,i] for j in range(n_rows) for i in range(n_train_cols) ]
    test_coords = [[j,i] for j in range(n_rows) for i in range(n_test_cols) ]

    bin_n_data = [bin_n for _ in train_cells]
    hog_data = [list(map(hog,row, bin_n_data)) for row in train_cells]
    X_train = np.float32(hog_data).reshape(-1,bin_n*4)
    X_train = preprocessing.normalize(X_train, norm='l2')
    y_train = np.repeat(categories,n_train_cols*n_rowsPerCategory)[:,np.newaxis]

    bin_n_data = [bin_n for _ in test_cells]
    hog_data = [list(map(hog,row, bin_n_data)) for row in test_cells]
    X_test = np.float32(hog_data).reshape(-1,bin_n*4)
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

# Indices of support vectors
    ref_SVs = clf.support_
    idx_obj = randint(0, len(X_test) - 1)
    x = X_test[idx_obj]

# Contextualized evaluation process
    topK = clf.predict_with_context([x])[0]
    position = 0
    prediction = topK[position]
    idx_proMISV = ref_SVs[prediction.eval.membership.reason]
    idx_conMISV = ref_SVs[prediction.eval.nonmembership.reason]

    class_name = get_article(prediction.class_name) + ' $\mathbf{' + prediction.class_name + '}$'
        
    print("Object: {0} predictedClass[{5}]: {4} actualClass: {1} classMISV+: {2} classMISV-: {3} rel_idxMISV+: {6} rel_idxMISV-: {7}".format(
            idx_obj, y_test[idx_obj][0],y_train[idx_proMISV][0],y_train[idx_conMISV][0], prediction.class_name, position,prediction.eval.mu_hat.misv_idx, prediction.eval.nu_hat.misv_idx))
         
    drawing = (test_cells[test_coords[idx_obj][0]][test_coords[idx_obj][1]])
    misvPro = (train_cells[train_coords[idx_proMISV][0]][train_coords[idx_proMISV][1]])
    misvCon = (train_cells[train_coords[idx_conMISV][0]][train_coords[idx_conMISV][1]])

    from matplotlib.colors import Colormap, ListedColormap

    cArray =  ['#ffffff', '#cccccc', '#999999', '#666666', '#333333', '#000000']
    cArrayInv =  ['#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff']
    cm = ListedColormap(cArray)
    cmInv = ListedColormap(cArrayInv)


    fig = plt.figure(figsize=(12, 5))

    h0 = plt.subplot2grid((5, 10), (0, 0), rowspan=2, colspan=2)
    h2 = plt.subplot2grid((5, 10), (0, 3), colspan=8)
    h3 = plt.subplot2grid((5, 10), (1, 3), colspan=8)

    h4 = plt.subplot2grid((5, 10), (2, 0), colspan=10)

    ax0 = plt.subplot2grid((5, 10), (3, 0))
    ax1 = plt.subplot2grid((5, 10), (3, 1), colspan=4)
    ax2 = plt.subplot2grid((5, 10), (3, 5))
    ax3 = plt.subplot2grid((5, 10), (3, 6), colspan=4)

    ax4 = plt.subplot2grid((5, 10), (4, 0), colspan=6)
    ax5 = plt.subplot2grid((5, 10), (4, 6))
    ax6 = plt.subplot2grid((5, 10), (4, 7), colspan=3)

    h0.axis('off')

    h2.axis('off')
    h3.axis('off')
    h4.axis('off')

    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')

    h0.imshow(drawing, cmap=cmInv, interpolation='none')
    h2.text(0, 0, 'Predicted class:', fontsize=10, style='italic', fontweight='bold')
    h3.text(0, .75, prediction.class_name , fontsize=14,  fontweight='bold')
    h4.text(0, 0, 'Explanation:', fontsize=10, style='italic', fontweight='bold')


#   Explanation with 'E'xample, 'C'ounterexample, 'B'oth
    explanationType = 'B'

    if explanationType == 'E':
        ax0.imshow(drawing, cmap=cm, interpolation='none')
        ax1.text(0, 0.5, 'should be {0} because it looks similar to'.format(class_name), wrap=True)
        ax2.imshow(misvPro, cmap=cm, interpolation='none')
        ax3.text(0, 0.5, ', which has been identified as {0}.'.format(class_name), wrap=True)
    elif explanationType == 'C':
        ax0.imshow(drawing, cmap=cm, interpolation='none')
        ax1.text(0, 0.5, 'should be {0}. However, it looks like '.format(class_name), wrap=True)
        ax2.imshow(misvCon, cmap=cm, interpolation='none')
        ax3.text(0, 0.5, ', which has not been identified as {0}.'.format(class_name), wrap=True)
    else:
        ax0.imshow(drawing, cmap=cm, interpolation='none')
        ax1.text(0, 0.5, 'should be {0} because it looks similar to'.format(class_name), wrap=True)
        ax2.imshow(misvPro, cmap=cm, interpolation='none')
        ax3.text(0, 0.5, ', which has been identified as {0}.'.format(class_name), wrap=True)
        ax4.text(0, 0.5, 'However, the digit could not be {0} because it also looks like '.format(class_name), wrap=True)
        ax5.imshow(misvCon, cmap=cm, interpolation='none')
        ax6.text(0, 0.5, ', which has not been recognized as such.', wrap=True)


   

    plt.savefig("{0}/explaining_prediction-digit_{1}.pdf".format(output_dir, idx_obj), dpi=300)
    plt.show()

import time
if __name__ == '__main__':
    start_time = time.time()
    run_test(output_dir, get_article)
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))