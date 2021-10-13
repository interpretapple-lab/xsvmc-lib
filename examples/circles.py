#!/usr/bin/env python
#
# This example shows how to use xSVMC to perform contextualized binary classifications.
# 
# The example is based on the ScikitLearn tutorial entitled 'Classifier comparison' 
# posted on 
#   https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#
# Keywords: xSVMC, binary classification, contextualized classification
#


import numpy as np
import matplotlib.pyplot as plt
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


    
from sklearn.datasets import make_circles
ds = make_circles(noise=0.1, factor=0.5, random_state=1)
X, y = ds

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, shuffle=True)


degree = 2.0
gamma = 1.0
coef0 = 0.0
kernel = 'poly'
k = 1

clf = xSVMC(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, decision_function_shape='ovo') 
clf.fit(X_train, y_train)
ref_SVs = clf.support_

predictions = clf.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# ---------
h = .02  # step size in the mesh
figure = plt.figure(figsize=(10, 5))

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

# just plot the dataset first
from matplotlib.colors import ListedColormap
cm = plt.cm.RdBu
cArray =  ['#ff0000', '#0000ff']
cm_bright = ListedColormap(cArray)

nrows = 2
ncols = 3
idxObject = 0
for row in range(nrows):
    for col in range(ncols):
        ax = plt.subplot(nrows, ncols, idxObject+1)
        
       
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # plot support vectors
        for idx0 in range(len(ref_SVs)):
            idx1 = ref_SVs[idx0]
            v1 = X_train[idx1]
            idxColor = y_train[idx1]
            ax.scatter([v1[0]], [v1[1]], s=80, c=[cArray[idxColor]], alpha=1.0)
            ax.text(v1[0] + 0.09, v1[1]-0.05, ('%d' % idx0), size=7, horizontalalignment='left', color='#030303')


        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        x = X_test[idxObject]
        x_class = clf.predict([x])
        x_influence = clf.decision_function([x])
        
        print("---")
        topK = clf.predict_with_context(x)
        prediction = topK[0]

        idxPositiveMISV = ref_SVs[prediction.eval.mu_hat.misv_idx]
        idxNegativeMISV = ref_SVs[prediction.eval.nu_hat.misv_idx]

        print("Method 1 - Object: {0} predictedClass: {1} influence: {2}".format(idxObject, x_class[0], x_influence[0]))
        print("Method 2 - Object: {0} predictedClass: {1} influence: {5} actualClass: {2} classMISV+: {3} classMISV-: {4}".format(
                       idxObject, prediction.class_name, y_test[idxObject],y_train[idxPositiveMISV],y_train[idxNegativeMISV], prediction.eval.buoyancy))
      
       
        # plot the positiveMisv
        positiveMisv = X_train[idxPositiveMISV]
        ax.scatter(positiveMisv[0], positiveMisv[1], s=100,
                linewidth=2, facecolors='none', edgecolors='k', marker='o')

        # plot the negativeMisv
        negativeMisv = X_train[idxNegativeMISV]
        ax.scatter(negativeMisv[0], negativeMisv[1], s=100,
               linewidth=2, facecolors='none', edgecolors='k', marker='o')

        # plot the object
        ax.scatter([x[0]], [x[1]], s=40, c=[cArray[prediction.class_name]], alpha=1.0)
        ax.scatter([x[0]], [x[1]], s=100, c=[cArray[prediction.class_name]], marker='x')

        idxObject = idxObject + 1


figure.subplots_adjust(left=.02, right=.98)
plt.show()




