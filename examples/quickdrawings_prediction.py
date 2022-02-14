#!/usr/bin/env python
#
# This example shows how to use xSVMC to perform contextualized predictions of the
# classes of drawings. 
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
from sklearn import preprocessing
from common import hog, ordinal, ordinalltx
import os, sys

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    print('inserting {!r} into sys.path'.format(_path_to_lib_))
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

def run_example(output_dir):
    # Params
    BINS_N = 16 
    RASTER_SZ = 28
    CELL_SZ = 7
    N_CELLS = int(RASTER_SZ/CELL_SZ) 

    categories = ['ambulance', 'alarmclock', 'binoculars', 'bulldozer', 'crocodile',
              'hamburger', 'broom', 'castle', 'knee', 'snake',  'submarine', 'sun', 
              'telephone' , 'trumpet' ,'violin', 'guitar' ]

    n_objectsPerRow = 100
    n_rowsPerCategory = 10
    training_ratio = 0.6

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

    train_coords = [[j,i] for j in range(n_rows) for i in range(n_train_cols) ]
    test_coords = [[j,i] for j in range(n_rows) for i in range(n_test_cols) ]

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
    gamma = 10.0
    coef0 = 0.0
    c0 =  0.1
    k = 2


    clf = xSVMC(kernel=kernel, C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
# Learning process
    print("Learning...")
    clf.fit(X_train, y_train.ravel())

# Indices of support vectors
    ref_SVs = clf.support_

    idx_obj = 0

    plotRows = k + 4
    plotCols = 4

    from matplotlib.colors import ListedColormap


    cArray =  ['#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff']
    cm = ListedColormap(cArray)
# cm = plt.cm.gray

    while idx_obj < len(test_coords):
        x = X_test[idx_obj]
        drawing = (test_cells[test_coords[idx_obj][0]][test_coords[idx_obj][1]])
       
        actual_class =  y_test[idx_obj][0]
    # Contextualized evaluation process
        topK = clf.predict_with_context(x)


        fig = plt.figure(figsize=(plotCols+6, plotRows))
        h0 = plt.subplot2grid((plotRows, plotCols), (0, 0), colspan=plotCols, rowspan=2)
        plt.title( 'obj_{0} ({1})'.format(idx_obj, actual_class), style='italic', weight='bold', ha='center', size='large' )
        h0.axis('off')

        sh0 = plt.subplot2grid((plotRows, plotCols), (3, 0))
        sh1 = plt.subplot2grid((plotRows, plotCols), (3, 1))
        sh2 = plt.subplot2grid((plotRows, plotCols), (3, 2))
        sh3 = plt.subplot2grid((plotRows, plotCols), (3, 3))
        sh0.axis('off')
        sh1.axis('off')
        sh2.axis('off')
        sh3.axis('off')

        h0.imshow(drawing, cmap=cm, interpolation='none')

        sh0.text(0.5, 0, 'Rank', style='italic', weight='demibold', ha='center', size='medium')
        sh1.text(0.5, 0, 'Predicted Class', style='italic', weight='demibold', ha='center', size='medium')
        sh2.text(0.5, 0, 'MISV+', style='italic', weight='demibold', ha='center', size='medium')
        sh3.text(0.5, 0, 'MISV-', style='italic', weight='demibold', ha='center', size='medium')


        plot_ctr = 0
        for position in range(k):
            prediction = topK[position]
            idx_proMISV = ref_SVs[prediction.eval.membership.reason]
            idx_conMISV = ref_SVs[prediction.eval.nonmembership.reason]
        
            print("Object: {0} {5}_predictedClass: {4} actualClass: {1} classMISV+: {2} classMISV-: {3} rel_idxMISV+: {6} rel_idxMISV-: {7}".format(
            idx_obj, y_test[idx_obj][0],y_train[idx_proMISV][0],y_train[idx_conMISV][0], prediction.class_name, ordinal(position + 1),prediction.eval.mu_hat.misv_idx, prediction.eval.nu_hat.misv_idx))
         
            misvPro = (train_cells[train_coords[idx_proMISV][0]][train_coords[idx_proMISV][1]])
            misvCon = (train_cells[train_coords[idx_conMISV][0]][train_coords[idx_conMISV][1]])

            ax0 = plt.subplot2grid((plotRows, plotCols), (position+4, plot_ctr))
            ax0.axis('off')
            ax0.text(0.5, 0.5, ordinalltx(position+1), style='italic', weight='roman', ha='center', size='small')
            plot_ctr+=1

            ax1 = plt.subplot2grid((plotRows, plotCols), (position+4, plot_ctr))
            ax1.axis('off')
            ax1.text(0.5, 0.5, prediction.class_name, style='italic', weight='roman', ha='center', size='small')
            plot_ctr+=1
        
            ax2 = plt.subplot2grid((plotRows, plotCols), (position+4, plot_ctr))
            ax2.axis('off')
            ax2.imshow(misvPro, cmap=cm, interpolation='none')
            plot_ctr+=1

            ax3 = plt.subplot2grid((plotRows, plotCols), (position+4, plot_ctr))
            ax3.axis('off')
            ax3.imshow(misvCon, cmap=cm, interpolation='none')
            plot_ctr=0


        plt.subplots_adjust(hspace=0.5)
        plt.savefig("{0}/top_{2}_predictions-quickdrawings_{1}.pdf".format(output_dir, idx_obj, k), dpi=300)
        #idx_obj+= 100
        idx_obj+= n_rowsPerCategory * n_test_cols
        plt.show()

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example(output_dir)
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))