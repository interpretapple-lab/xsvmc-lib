#
#  This is an implementation of a measure including the "notion of complement" proposed in [1] to compute 
#  the similarity between two intuitionistic fuzzy sets. 
#
#  References:
#
#  [1] Szmidt, E., Kacprzyk, J.: A concept of similarity for intuitionistic fuzzy sets and its use 
#      in group decision making. In: IEEE International Conference on Fuzzy Systems, pp. 1129–1134 (2004)

import os, sys



_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xsvmlib.xmodels import xAIFSElement

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

def f(a, b):
    ret = a / (a + b)
    return ret
        

def complement(A):
    cA = [xAIFSElement(A[i].nonmembership, A[i].membership) for i in range(len(A))]
    return cA


def distance(A,B):
    ret = 0.0
    nA = len(A)
    nB = len(B)
    if nA != nB:
        raise Exception("Length of A is not equal to length of B.")

    sum = 0.0
    n = nA

    for i in range(n):
        elemA = A[i]
        elemB = B[i] 
        sum = sum + abs(elemA.mu_hat.value - elemB.mu_hat.value) + abs(elemA.nu_hat.value - elemB.nu_hat.value) + abs(elemA.hesitation - elemB.hesitation)
    
    ret = 0.5 * sum / n
    return ret


def similarity(A,B):
    ret = 0.5
    nA = len(A)
    nB = len(B)
    if nA != nB:
        raise Exception("Length of A is not equal to length of B.")

    cB = complement(B)
    sum = 0

    d1 = distance(A, B)
    d2 = distance(A, cB)

    if (d1 + d2 > 0):
        ret = (1 - f(d1, d2)) / (1 + f(d1, d2))
    
    return ret