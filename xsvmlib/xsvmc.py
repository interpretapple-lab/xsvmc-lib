from sklearn.svm._classes import SVC 
from xsvmlib.xmodels import xAAD, xAIFSElement, xPrediction
import numpy as np
from math import exp



class xSVMC(SVC):
    """Explainable Support Vector Machine Classification
    
    Implementation based on Scikit-learn SVC class,
    with a modified version of the decision and predict
    functions which identifies the MISV for each class

   Parameters
    ----------
        
    k: int, default=1
        Number of possible classes expected for the
        prediction output.
        
    """
        
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovo",
        break_ties=False,
        random_state=None,
        k = 1
    ):

        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

        if(not isinstance(k, int)):
            raise ValueError("K parameter must be an integer")
        elif(k < 1):
            raise ValueError("K parameter cannot lower than 0")
        self.k = k

    def compute_kernel(self,V, X, params = None):
        """ Computes the kernel between vectors V and X
            Currently implemented kernels:
            - Linear
            - Polynomial
            - Radial basis function (rbf)
            - Custom kernel function
        
        """
        if params is None:
            params = self.get_params()

        if self.kernel == 'linear':
            return np.dot(V, X)
        elif self.kernel == 'poly':
            return ((params['gamma']*np.dot(V, X)) + params['coef0'])**params['degree']
        elif self.kernel == 'rbf':
            return exp(-params['gamma'] * np.dot(V - X, V - X)) 
        elif callable(self.kernel):
            return self.kernel(V, X) 

        

    def decision_function_with_context(self, X):
        """ Evaluates the decision function for the sample X.

        Parameters
        X: ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns
        df: ndarray of shape (n_classes * (n_classes-1) / 2,).
            Returns the decision function of the sample for each class in the model. Since decision_function_shape='ovo'
            is always used as multi-class strategy, df is an array of shape (n_classes * (n_classes-1) / 2,).

        Notes
            About decision_function_shape : {'ovo', 'ovr'}

            Whether to return a one-vs-rest ('ovr') decision function of shape
            (n_samples, n_classes) as all other classifiers, or the original
            one-vs-one ('ovo') decision function of libsvm which has shape
            (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
            ('ovo') is always used as multi-class strategy. The parameter is
            ignored for binary classification.

            See https://github.com/scikit-learn/scikit-learn/blob/23afd5d95c18915c55070cecaecf9f3030ae9bbb/sklearn/svm/_classes.py
                    
        """

        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        if len(self.classes_) == 2:
            return self.__binary_decision_function_with_context(X)

        nv = self.n_support_
        a = self._dual_coef_
        b  = self._intercept_
        sv = self.support_vectors_
       
        
        params = self.get_params()

        kernel_values = [self.compute_kernel(v, X, params) for v in sv] 

        start = [sum(nv[:i]) for i in range(len(nv))]
        end = [start[i] + nv[i] for i in range(len(nv))]
        
        xIFSElements = {}
        for i in range(len(b)):
            xIFSElements[i] = xAIFSElement()

        p2 = 0
        
        for i in range(len(nv)):
            for j in range(i+1,len(nv)):
                influence_pro_i = 0
                influence_pro_j = 0

                max_influence_pro_i = xAAD()
                max_influence_pro_j = xAAD()

                for p in range(start[i], end[i]):
                    value = a[j-1][p] * kernel_values[p]
                    if value >= 0:
                        influence_pro_i += value
                        if value >= max_influence_pro_i.value:
                            max_influence_pro_i.value = value
                            max_influence_pro_i.misv_idx = p
                    else:
                        influence_pro_j -= value
                        if abs(value) >= max_influence_pro_j.value:
                            max_influence_pro_j.value = abs(value)
                            max_influence_pro_j.misv_idx = p

                    #print("i: {0}, j: {1}, sv: {2}, value: {8:3.4f}, max_pro_i: {3:3.4f}, {4:3.4f}, max_con_i: {5:3.4f}, {6},  a[j-1][p]: {7:3.4f}".format(i,j,p, max_influence_pro_i.value, max_influence_pro_i.misv_idx, max_influence_pro_j.value, max_influence_pro_j.misv_idx, a[j-1][p], value))

                for p in range (start[j], end[j]):
                    value = a[ i ][p] * kernel_values[p]
                    if value >= 0:
                        influence_pro_i += value
                        if value >= max_influence_pro_i.value:
                            max_influence_pro_i.value = value
                            max_influence_pro_i.misv_idx = p
                    else:
                        influence_pro_j -= value
                        if abs(value) >= max_influence_pro_j.value:
                            max_influence_pro_j.value = abs(value)
                            max_influence_pro_j.misv_idx = p

                    #print("i: {0}, j: {1}, sv: {2}, value: {8:3.4f}, max_pro_i: {3:3.4f}, {4}, max_con_i: {5:3.4f}, {6}, a[i][p]: {7:3.4f} ".format(i,j,p, max_influence_pro_i.value, max_influence_pro_i.misv_idx, max_influence_pro_j.value, max_influence_pro_j.misv_idx, a[i][p], value))

                
               
                if b[p2]>0:
                    influence_pro_i+=b[p2]
                else:
                    influence_pro_j-=b[p2]

                xIFSElements[p2].mu_hat = xAAD(influence_pro_i, max_influence_pro_i.misv_idx) 
                xIFSElements[p2].nu_hat = xAAD(influence_pro_j, max_influence_pro_j.misv_idx)

                # print("i: {5} j: {6} pro_i: {0:3.4f}, pro_j: {1:3.4f}, buoyancy: {2:3.4f}, pro_i.misv_idx: {3},  pro_j.misv_idx: {4}".format(xIFSElements[p2].mu_hat.value, xIFSElements[p2].nu_hat.value, xIFSElements[p2].buoyancy, xIFSElements[p2].mu_hat.misv_idx, xIFSElements[p2].nu_hat.misv_idx, i, j))


                p2 += 1

        return xIFSElements


    def __binary_decision_function_with_context(self, X):
        """ Evaluates the binary decision function for the sample X.

        Parameters
        X: ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns
        df: ndarray of shape (n_classes * (n_classes-1) / 2,).
            Returns the decision function of the sample for each class in the model. Since decision_function_shape='ovo'
            is always used as multi-class strategy, df is an array of shape (n_classes * (n_classes-1) / 2,).

        """


        nv = self.n_support_
        a = self._dual_coef_
        b  = self._intercept_
        sv = self.support_vectors_
        
        params = self.get_params()

        kernel_values = [self.compute_kernel(v, X, params) for v in sv] 

        start = [sum(nv[:i]) for i in range(len(nv))]
        end = [start[i] + nv[i] for i in range(len(nv))]
        
        xIFSElements = {}
        for i in range(len(b)):
            xIFSElements[i] = xAIFSElement()

        p2 = 0
        
        for i in range(len(nv)):
            for j in range(i+1,len(nv)):
                influence_pro_i = 0
                influence_con_i = 0

                max_influence_pro_i = xAAD()
                max_influence_con_i = xAAD()

                for p in range(start[i], end[i]):
                    value = a[j-1][p] * kernel_values[p]
                    if value >= 0:
                        influence_pro_i += value
                        if value >= max_influence_pro_i.value:
                            max_influence_pro_i.value = value
                            max_influence_pro_i.misv_idx = p
                    else:
                        influence_con_i -= value
                        if abs(value) >= max_influence_con_i.value:
                            max_influence_con_i.value = abs(value)
                            max_influence_con_i.misv_idx = p

                    # print("i: {0}, j: {1}, sv: {2}, value: {8:3.4f}, max_pro_i: {3:3.4f}, {4:3.4f}, max_con_i: {5:3.4f}, {6},  a[j-1][p]: {7:3.4f}".format(i,j,p, max_influence_pro_i.value, max_influence_pro_i.misv_idx, max_influence_con_i.value, max_influence_con_i.misv_idx, a[j-1][p], value))

                for p in range (start[j], end[j]):
                    value = a[ i ][p] * kernel_values[p]
                    if value >= 0:
                        influence_pro_i += value
                        if value >= max_influence_pro_i.value:
                            max_influence_pro_i.value = value
                            max_influence_pro_i.misv_idx = p
                    else:
                        influence_con_i -= value
                        if abs(value) >= max_influence_con_i.value:
                            max_influence_con_i.value = abs(value)
                            max_influence_con_i.misv_idx = p

                    # print("i: {0}, j: {1}, sv: {2}, value: {8:3.4f}, max_pro_i: {3:3.4f}, {4}, max_con_i: {5:3.4f}, {6}, a[i][p]: {7:3.4f} ".format(i,j,p, max_influence_pro_i.value, max_influence_pro_i.misv_idx, max_influence_con_i.value, max_influence_con_i.misv_idx, a[i][p], value))

                
               
                if b[p2]>0:
                    influence_pro_i+=b[p2]
                else:
                    influence_con_i-=b[p2]

                # Consider the evaluation of the proposition 'X IS A'. In the implementation of SVC, while SVs favoring the propositon 
                # are related to self.classes_[1], SVs against the propositon are related to self.classes_[0].
                # Since i=0 and the influence influence_pro_i has been computed below (i.e., evaluation of 'X IS NOT A'), the xIFSElements is built 
                # as follows: 
                xIFSElements[p2].mu_hat = xAAD(influence_con_i, max_influence_con_i.misv_idx)
                xIFSElements[p2].nu_hat = xAAD(influence_pro_i, max_influence_pro_i.misv_idx) 
                
                # print("mu: {0:3.4f}, nu: {1:3.4f}, buoyancy: {2:3.4f}, mu.misv_idx: {3},  nu.misv_idx: {4}".format(xIFSElements[p2].mu_hat.value, xIFSElements[p2].nu_hat.value, xIFSElements[p2].buoyancy, xIFSElements[p2].mu_hat.misv_idx, xIFSElements[p2].nu_hat.misv_idx))

                p2 += 1

        return xIFSElements

    def __sort_fn(self, e):
        return e[1]

    def predict_with_context(self, X):
        """Performs an augmented prediction of the top-K classes for the sample X 
        
        Parameters
        X :  ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns
        topK : list of the top-K classes predicted for X.

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        cs = self.classes_
        k = self.k
        
        ifselements = self.decision_function_with_context(X)

        if len(cs)==2:
            if ifselements[0].buoyancy > 0: 
                return [xPrediction(cs[1],ifselements[0])]
            else:
                return [xPrediction(cs[0],ifselements[0])]

        votes = [0 for c in range(len(cs))]
        evals = [xAIFSElement() for c in range(len(cs))]

        p = 0
        for i in range(len(cs)):
            for j in range(i+1,len(cs)):
                if ifselements[p].buoyancy > 0: # pro_i 
                    votes[i]+=1
                else: # pro_j 
                    votes[j]+=1

                if ifselements[p].mu_hat.value >= evals[i].mu_hat.value:
                    evals[i].mu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= evals[i].nu_hat.value:
                    evals[i].nu_hat = ifselements[p].nu_hat

                if ifselements[p].mu_hat.value >= evals[j].nu_hat.value:
                    evals[j].nu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= evals[j].mu_hat.value:
                    evals[j].mu_hat = ifselements[p].nu_hat
                                  
                p+=1

                # print("votes[{0}]: {2} votes[{1}]: {3} evals[{0}]: {4:5.4f}, {5}, {6} evals[{1}]: {7:5.4f}, {8}, {9}".format(
                #     i,j,votes[i],votes[j], evals[i].buoyancy, evals[i].mu_hat.misv_idx, evals[i].nu_hat.misv_idx,
                #     evals[j].buoyancy, evals[j].mu_hat.misv_idx, evals[j].nu_hat.misv_idx ))
        
        indices_classes =  [c for c in range(len(cs))]  
        sorted_classes = list(zip(indices_classes, votes, evals))
        sorted_classes.sort(key=self.__sort_fn,reverse=True)

        # Return collection of xPrediction
        topK = [xPrediction(cs[t[0]],t[2]) for t in sorted_classes[0:k]]
        return topK


    def  is_member_of(self, X, class_idx):
        """Performs an augmented evaluation of 'X IS A', where A is the class referenced by class_idx
        
        Parameters
        X:  ndarray of shape (n_features,) consisting of n features identified for X. 
        class_idx: class index

        Returns
        elem : Augmented IFSElement representing the augmented evaluation of 'X IS A'.
        """

        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        cs = self.classes_
        k = self.k
        
        ifselements = self.decision_function_with_context(X)

        if len(cs)==2:
            ifselements[0]

        evals = [xAIFSElement() for c in range(len(cs))]

        p = 0
        for i in range(len(cs)):
            for j in range(i+1,len(cs)):
                if ifselements[p].mu_hat.value >= evals[i].mu_hat.value:
                    evals[i].mu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= evals[i].nu_hat.value:
                    evals[i].nu_hat = ifselements[p].nu_hat

                if ifselements[p].mu_hat.value >= evals[j].nu_hat.value:
                    evals[j].nu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= evals[j].mu_hat.value:
                    evals[j].mu_hat = ifselements[p].nu_hat
                                  
                p+=1

        return evals[class_idx]