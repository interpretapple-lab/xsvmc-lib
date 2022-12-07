#!/usr/bin/env python
#
# This example shows how to use the most influential support vectors to build a visual representation 
# of a contextualized evaluation. 
#
# Keywords: xSVMC, multiclass classification, contextualized classification, visual representation
#

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from common import hog, deskew, get_imgWithInfluence, get_influenceMap
from sklearn import preprocessing
import os, sys

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
    RASTER_SZ = 28
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
    k = 1


    clf = xSVMC(kernel=kernel, C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
# Learning process
    print("Learning...")
    clf.fit(X_train, y_train.ravel())

# Indices of support vectors
    ref_SVs = clf.support_
    SVs = clf.support_vectors_

    idx_obj = 0

    plotRows = k
    plotCols = 4

    from matplotlib.colors import ListedColormap

    cArray =  ['#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff']
    cm = ListedColormap(cArray)

    BG_COLOR = (233,233,233)
    while idx_obj < len(test_coords):
        plt.figure(figsize=(plotCols*3, plotRows*3))
        x = X_test[idx_obj]
        object = deskew(test_cells[test_coords[idx_obj][0]][test_coords[idx_obj][1]], raster_sz=RASTER_SZ)
        object_gradients = get_imgWithInfluence(object, raster_sz=RASTER_SZ, ret_img_sz = 128, bg_color=BG_COLOR)
           
    # Contextualized evaluation process
        topK = clf.predict_with_context([x])[0]

        plot_ctr = 1
        for position in range(k):
            prediction = topK[position]
            idx_proMISV = ref_SVs[prediction.eval.mu_hat.misv_idx]
            idx_conMISV = ref_SVs[prediction.eval.nu_hat.misv_idx]
        
            print("Object: {0} predictedClass[{5}]: {4} actualClass: {1} classMISV+: {2} classMISV-: {3} rel_idxMISV+: {6} rel_idxMISV-: {7}".format(
            idx_obj,y_test[idx_obj][0],y_train[idx_proMISV][0],y_train[idx_conMISV][0], prediction.class_name, position,prediction.eval.mu_hat.misv_idx, prediction.eval.nu_hat.misv_idx))
         
            misvPro = deskew(train_cells[train_coords[idx_proMISV][0]][train_coords[idx_proMISV][1]], raster_sz=RASTER_SZ)
            misvCon = deskew(train_cells[train_coords[idx_conMISV][0]][train_coords[idx_conMISV][1]], raster_sz=RASTER_SZ)
              
            misvPro_gradients = get_imgWithInfluence(misvPro, raster_sz=RASTER_SZ, ret_img_sz = 128, fg_color = (32,133,64), bg_color=BG_COLOR)
            misvCon_gradients = get_imgWithInfluence(misvCon, raster_sz=RASTER_SZ, ret_img_sz = 128, fg_color = (0, 113, 188), bg_color=BG_COLOR)
            imgWithInfluence = get_influenceMap(object, x, SVs[prediction.eval.mu_hat.misv_idx], SVs[prediction.eval.nu_hat.misv_idx], raster_sz = RASTER_SZ, ret_img_sz = 128,  bins_n = 16, bg_color=BG_COLOR)

            
            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(object_gradients, cmap=cm, interpolation='none'),plt.axis('off')
            plt.title('obj_{0} ({1})'.format(idx_obj, y_test[idx_obj][0] ), style='normal', weight='normal', ha='center', size='x-small')
            plot_ctr+=1
       
            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(misvPro_gradients, cmap=cm, interpolation='none'),plt.axis('off')
            plt.title('MISV+ ({0})'.format(y_train[idx_proMISV][0]), style='normal', weight='normal', ha='center', size='x-small')
            plot_ctr+=1

            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(misvCon_gradients, cmap=cm, interpolation='none'),plt.axis('off')
            plt.title('MISV- ({0})'.format(y_train[idx_conMISV][0]), style='normal', weight='normal', ha='center', size='x-small')
            plot_ctr+=1

            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(imgWithInfluence),plt.axis('off')
            plt.title('Influence Map', style='normal', weight='normal', ha='center', size='x-small')
            plot_ctr+=1
        plt.subplots_adjust(hspace=0.5)
        plt.savefig('output/quickdrawing_influence_{0}.pdf'.format(idx_obj))
        plt.show()

        idx_obj+= 125

import time
if __name__ == '__main__':
    start_time = time.time()
    run_example()
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))
