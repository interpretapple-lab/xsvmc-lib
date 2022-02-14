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

def run_test():
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
    y_train = np.repeat(np.arange(10),250)[:,np.newaxis]

    raster_sz_data = [RASTER_SZ for _ in test_cells]
    deskewed = [list(map(deskew,row, raster_sz_data)) for row in test_cells]
    bin_n_data = [BINS_N for _ in deskewed]
    hog_data = [list(map(hog,row, bin_n_data)) for row in deskewed]

    X_test = np.float32(hog_data).reshape(-1,BINS_N*4)
    y_test =  np.repeat(np.arange(10),250)[:,np.newaxis]

    degree = 2.0
    gamma = 5.383
    coef0 = 0.0
    c0 =  2.67
    k = 1


    clf = xSVMC(kernel='poly', C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
    clf.fit(X_train, y_train.ravel())
    ref_SVs = clf.support_
    SVs = clf.support_vectors_

    idx_obj = 0

    plotRows = k
    plotCols = 4

    from matplotlib.colors import Colormap, ListedColormap

    cArray =  ['#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff']
    cm = ListedColormap(cArray)

    
    BG_COLOR = (233,233,233)
    while idx_obj < len(digits_coord):
        plt.figure(figsize=(plotCols*3, plotRows*3))
        x = X_test[idx_obj]
        digit = deskew(test_cells[digits_coord[idx_obj][0]][digits_coord[idx_obj][1]], raster_sz=RASTER_SZ)
        #object_gradients = write_imgWithInfluence(digit,'output/digits_influence_object-{0}.pdf'.format(idx_obj), raster_sz=RASTER_SZ, ret_img_sz = 128, fg_color = (0, 0, 0), bg_color=BG_COLOR)
        # object_gradients = write_imgWithPixels(digit,'output/digits_influence_object-{0}.pdf'.format(idx_obj), raster_sz=RASTER_SZ, ret_img_sz = 200, fg_color = (0, 0, 0), bg_color=(255,255,255))
        object_gradients = get_imgWithInfluence(digit, raster_sz=RASTER_SZ, ret_img_sz = 128, fg_color = (0, 0, 0), bg_color=(255,255,255))
     
    # Contextualized evaluation process
        topK = clf.predict_with_context(x)

        plot_ctr = 1
        for position in range(k):
            prediction = topK[position]
            idx_proMISV = ref_SVs[prediction.eval.mu_hat.misv_idx]
            idx_conMISV = ref_SVs[prediction.eval.nu_hat.misv_idx]
        
            print("Object: {0} predictedClass[{5}]: {4} actualClass: {1} classMISV+: {2} classMISV-: {3} rel_idxMISV+: {6} rel_idxMISV-: {7}".format(
            idx_obj,y_test[idx_obj][0],y_train[idx_proMISV][0],y_train[idx_conMISV][0], prediction.class_name, position,prediction.eval.mu_hat.misv_idx, prediction.eval.nu_hat.misv_idx))
         
            misvPro = deskew(train_cells[digits_coord[idx_proMISV][0]][digits_coord[idx_proMISV][1]], raster_sz=RASTER_SZ)
            misvCon = deskew(train_cells[digits_coord[idx_conMISV][0]][digits_coord[idx_conMISV][1]], raster_sz=RASTER_SZ)

            misvPro_gradients = get_imgWithInfluence(misvPro, raster_sz=RASTER_SZ, ret_img_sz = 128, fg_color = (32,133,64), bg_color=(255,255,255))
            misvCon_gradients = get_imgWithInfluence(misvCon, raster_sz=RASTER_SZ,ret_img_sz = 128, fg_color = (0, 113, 188), bg_color=(255,255,255))
            imgWithInfluence = get_influenceMap(digit,  x, SVs[prediction.eval.mu_hat.misv_idx], SVs[prediction.eval.nu_hat.misv_idx], raster_sz = 20, ret_img_sz = 128,  bins_n = 16)

            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(object_gradients, cmap=cm),plt.axis('off')
            plt.title('obj_{0} ({1})'.format(idx_obj, y_test[idx_obj][0] ), style='normal', weight='normal', ha='center', size='small')
            plot_ctr+=1
       
            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(misvPro_gradients, cmap=cm),plt.axis('off')
            plt.title('MISV+ ({0})'.format(y_train[idx_proMISV][0]), style='normal', weight='normal', ha='center', size='small')
            plot_ctr+=1

            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(misvCon_gradients, cmap=cm),plt.axis('off')
            plt.title('MISV- ({0})'.format(y_train[idx_conMISV][0]), style='normal', weight='normal', ha='center', size='small')
            plot_ctr+=1

            plt.subplot(plotRows,plotCols,plot_ctr),plt.imshow(imgWithInfluence),plt.axis('off')
            plt.title('Influence Map', style='normal', weight='normal', ha='center', size='small')
            plot_ctr+=1
    
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        idx_obj+= 125

import time
if __name__ == '__main__':
    start_time = time.time()
    run_test()
    print ('\nTest done! Time taken: {0:.3f} seconds.'.format(time.time() - start_time))
