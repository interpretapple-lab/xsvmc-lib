from matplotlib import pyplot as plt
import numpy as np
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


def plot_hyperplane(clf, cls_idx, min_x, max_x, linestyle, label,  color):
    # get the separating hyperplane
    w = clf.coef_[cls_idx]
    if w[1]>0:
        a = -w[0] / w[1]
        xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
        yy = a * xx - (clf.intercept_[cls_idx]) / w[1]
        plt.plot(xx, yy, linestyle, label=label, markerfacecolor=color, linewidth=1)
    else:
        yy = np.linspace(min_x - 5, max_x + 5)
        xx = np.repeat(-w[0], len(yy))
        plt.plot(xx, yy, linestyle, label=label, markerfacecolor=color, linewidth=1)
    plt.legend()

def plot_subfigure(clf, X, Y, x, predictionOrder, predictedClass, positiveMISV, negativeMISV):
     
    X = np.array(X)
    Y = np.array(Y)

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    if predictionOrder>2:
        print('Adjustment of plot layout is needed')
        return
    titlePart = ''
    if predictionOrder == 1:
        titlePart = 'Best prediction: '
    if predictionOrder == 2:
        titlePart = 'Second best prediction: '


    plt.subplot(1, 2, predictionOrder)
    plt.title(titlePart + predictedClass)
   
    

    a_class = np.where(Y=='A')
    b_class = np.where(Y=='B')
    c_class = np.where(Y=='C')
    d_class = np.where(Y=='D')
   
    plt.scatter(
        X[a_class, 0],
        X[a_class, 1],
        s=80,
        edgecolors="green",
        facecolors="none",
        linewidths=1,
        label="A",
        marker='h'
    )
    plt.scatter(
        X[b_class, 0],
        X[b_class, 1],
        s=40,
        edgecolors="indigo",
        facecolors="none",
        linewidths=1,
        label="B",
        marker="^"
    )
    plt.scatter(
        X[c_class, 0],
        X[c_class, 1],
        s=40,
        edgecolors="darkblue",
        facecolors="none",
        linewidths=1,
        label="C",
        marker='s'
    )
    plt.scatter(
        X[d_class, 0],
        X[d_class, 1],
        s=40,
        edgecolors="darkcyan",
        facecolors="none",
        linewidths=1,
        label="D",
        marker='D'
    )


    plt.scatter(
            clf.support_vectors_[:,0],
            clf.support_vectors_[:,1],
            s=140,
            edgecolors="black",
            facecolors="none",
            linewidths=1,
            label="SVs",
            marker='o'
        )

    plt.scatter(
            x[0],
            x[1],
            s=80,
            edgecolors="black",
            facecolors="gray",
            linewidths=1,
            label="x",
            marker='*'
        )
    
    plt.scatter(
            positiveMISV[0],
            positiveMISV[1],
            s=40,
            facecolors="darkgreen",
            linewidths=1,
            label="MISV+",
            marker="2"
        )
    plt.scatter(
            negativeMISV[0],
            negativeMISV[1],
            s=40,
            facecolors="darkred",
            linewidths=1,
            label="MISV-",
            marker="1"
        )



    if predictedClass in ['A', 'B']:
        plot_hyperplane(
            clf, 0,  min_x, max_x, "k--", "A vs B", 'b'
        )
    if predictedClass in ['A', 'C']:    
        plot_hyperplane(
            clf, 1,  min_x, max_x, "k-.", "A vs C", 'm'
        )
    if predictedClass in ['A', 'D']:
        plot_hyperplane(
            clf, 2,  min_x, max_x, "k:", "A vs D", 'darkgrey'
        )
    if predictedClass in ['B', 'C']:
        plot_hyperplane(
            clf, 3,  min_x, max_x, "k-.", "B vs C", 'b'
        )
    if predictedClass in ['D', 'B']:
        plot_hyperplane(
            clf, 4,  min_x, max_x, "k-.", "B vs D", 'b'
        )
    if predictedClass in ['C', 'D']:
        plot_hyperplane(
            clf, 5,  min_x, max_x, "k--", "C vs D", 'b'
        )

    plt.xlim(min_x - 0.5 * max_x, max_x + 0.5 * max_x)
    plt.ylim(min_y - 0.5 * max_y, max_y + 0.5 * max_y)




# Training data
X_train = [[2.5,4],[3.5, 4.5],[4,2.5],
    [-2.5,2.5],[-3, 4],[-4, 2.5],
    [-1.5,-2],[-2.5,-3.5],[-2,-5],
    [1.5,-3],[4,-4],[4.5,-4.5]]
y_train = ['A','A','A',
    'B','B','B', 
    'C','C','C',
    'D','D','D']
# Test data
x = [1,-1]
# Usage of XSVMC-Lib
from xsvmlib.xsvmc import xSVMC
clf = xSVMC(kernel='linear', k = 2)
# Learning process
clf.fit(X_train,y_train) 
# Access to SVM knowledge model
SVs = clf.support_vectors_
# Evaluation process
topK = clf.predict_with_context(x) 
# Access to contextualized predictions 
for pred in topK:
   print(pred.class_name,
      pred.eval.mu_hat.level,
      pred.eval.mu_hat.misv_idx,
      pred.eval.nu_hat.level,
      pred.eval.nu_hat.misv_idx,
      pred.eval.buoyancy)

# Visualization of contextualized predictions 
plt.figure(figsize=(12, 6))
i = 1
for pred in topK:
    plot_subfigure(clf, X_train, y_train, x, i, pred.class_name, SVs[pred.eval.mu_hat.misv_idx], SVs[pred.eval.nu_hat.misv_idx])
    i+=1
plt.subplots_adjust(wspace=0.1)
plt.savefig('output/simple_example.pdf')
plt.show()
