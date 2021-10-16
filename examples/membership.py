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
from matplotlib import pyplot as plt
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


SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


digits = cv.imread(cv.samples.findFile('examples/digits.png'),0)
if digits is None:
    raise Exception("we need the digits.png image from samples/data here !")

cells = [np.hsplit(row,100) for row in np.vsplit(digits,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

digits_coord = [[i,j] for i in range(50) for j in range(50)]

deskewed = [list(map(deskew,row)) for row in train_cells]
hog_data = [list(map(hog,row)) for row in deskewed]

X_train = np.float32(hog_data).reshape(-1,64)
y_train = np.repeat(np.arange(10),250)[:,np.newaxis]

deskewed = [list(map(deskew,row)) for row in test_cells]
hog_data = [list(map(hog,row)) for row in deskewed]

X_test = np.float32(hog_data).reshape(-1,bin_n*4)
y_test =  np.repeat(np.arange(10),250)[:,np.newaxis]

degree = 2.0
gamma = 5.383
coef0 = 0.0
c0 =  2.67
k = 3

clf = xSVMC(kernel='poly', C=c0, degree=degree, gamma=gamma, coef0=coef0, k = k)
clf.fit(X_train, y_train.ravel())
ref_SVs = clf.support_

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
