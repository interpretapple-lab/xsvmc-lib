from sklearn.svm import SVC 
from xsvmlib.xmodels import xAAD, xAIFSElement, xPrediction
import numpy as np
from math import exp


class xSVMC(SVC):
    """Explainable Support Vector Machine Classification
    
    This class is an implementation of the variant of the *support vector machine* (SVM)[1] classification process, 
    called *explainable SVM classification* (XSVMC), proposed in [2]. In XSVMC the most influential support vectors (MISVs) 
    are used for identifying what has been relevant to the classification. These MISVs can be used for contextualizing the 
    evaluations in such a way that the forthcoming predictions can be explained with ease.
    
    This implementation is based on Scikit-learn SVC class.
  
    Parameters:

    k: int, default=1
        Number of possible classes expected for the
        prediction output.

    References:
        [1] V.N.Vapnik,The Nature of Statistical Learning Theory, Springer-Verlag, New York, NY, USA, 1995.
            http://dx.doi.org/10.1007/978-1-4757-3264-1

        [2] M. Loor and G. De Tré. Contextualizing Support Vector Machine Predictions.
            International Journal of Computational Intelligence Systems, Volume 13, Issue 1, 2020,
            Pages 1483-1497,  ISSN 1875-6883, https://doi.org/10.2991/ijcis.d.200910.002

        
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

        
    def __decision_function_with_context_1dim(self, X):
        """ Evaluates the decision function for sample X.

        Parameters:
        X: ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns:
        df: ndarray of shape (n_classes * (n_classes-1) / 2,).
            Returns the decision function of the sample for each class in the model. Since decision_function_shape='ovo'
            is always used as multi-class strategy, df is an array of shape (n_classes * (n_classes-1) / 2,).

        Notes:
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
            return self.__binary_decision_function_with_context_1dim(X)

        nv = self.n_support_ # number of SVs for each class
        n_classes = len(nv)
        a = self._dual_coef_
        b  = self._intercept_
        sv = self.support_vectors_
       
        
        params = self.get_params()

        kernel_values = [self.compute_kernel(v, X, params) for v in sv] 
        
        start = [ sum(nv[:i]) for i in range(n_classes)]
        # end = [start[i] + nv[i] for i in range(len(nv))]
        end = start + nv
       
        
        xIFSElements = {}
        # for i in range(len(b)):
        #     xIFSElements[i] = xAIFSElement()

        p2 = 0
        for i in range(n_classes):
            for j in range(i+1,n_classes):
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
                        influence_pro_j += abs(value)
                        if p in range (start[j], end[j]):
                            if abs(value) >= max_influence_pro_j.value:
                                max_influence_pro_j.value = abs(value)
                                max_influence_pro_j.misv_idx = p
             
                    # print("i: {0}, j: {1}, sv: {2}, max_pro_{0}: ({3:3.4f}, {4}), max_pro_{1}: ({5:3.4f}, {6}), value: {7:3.4f}".format(
                    #     i,j,p, max_influence_pro_i.value, max_influence_pro_i.misv_idx, max_influence_pro_j.value, max_influence_pro_j.misv_idx, value))

                for p in range (start[j], end[j]):
                    value = a[ i ][p] * kernel_values[p]
                    if value >= 0:
                        influence_pro_i += value
                        if p in range(start[i], end[i]):
                            if value >= max_influence_pro_i.value:
                                max_influence_pro_i.value = value
                                max_influence_pro_i.misv_idx = p
                    else:
                        influence_pro_j += abs(value)
                        if abs(value) >= max_influence_pro_j.value:
                            max_influence_pro_j.value = abs(value)
                            max_influence_pro_j.misv_idx = p

                    # print("i: {0}, j: {1}, sv: {2}, max_pro_{0}: ({3:3.4f}, {4}), max_pro_{1}: ({5:3.4f}, {6}), value: {7:3.4f}".format(
                    #     i,j,p, max_influence_pro_i.value, max_influence_pro_i.misv_idx, max_influence_pro_j.value, max_influence_pro_j.misv_idx, value))

               
                if b[p2]>0:
                    influence_pro_i+=b[p2]
                else:
                    influence_pro_j-=b[p2]

                xIFSElements[p2] = xAIFSElement(
                         xAAD(influence_pro_i, max_influence_pro_i.misv_idx)  ,
                         xAAD(influence_pro_j, max_influence_pro_j.misv_idx)
                )

                # print("-- {0}  pro_{1}: ({2:3.4f}, {3}), pro_{4}: ({5:3.4f}, {6} )".format(
                #     p2, i, xIFSElements[p2].mu_hat.value, xIFSElements[p2].mu_hat.misv_idx, 
                #         j, xIFSElements[p2].nu_hat.value, xIFSElements[p2].nu_hat.misv_idx))

                p2 += 1

        return xIFSElements

    def __decision_function_with_context_ndim(self, X):
        """ Evaluates the decision function for each sample in X.

        Parameters:
        X: ndarray of shape (n_samples, n_features). 

        Returns:
        df: ndarray of shape (n_samples, n_classes * (n_classes-1) / 2).
            Returns the decision function of the sample for each class in the model. Since decision_function_shape='ovo'
            is always used as multi-class strategy, df is an array of shape (n_classes * (n_classes-1) / 2,).

        Notes:
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

       
        nv = self.n_support_ # number of SVs for each class
        n_classes = len(nv)
        a = self._dual_coef_
        b  = self._intercept_
        sv = self.support_vectors_
        n_samples = X.shape[0]
        
        params = self.get_params()

        kernel_values = self.compute_kernel(X, np.transpose(sv), params)

        start = [ sum(nv[:i]) for i in range(n_classes)]
        end = np.add.accumulate(nv)
       
        weighted_vals = np.multiply( a, kernel_values[:, np.newaxis])

        n_comparisons = n_classes*(n_classes-1)//2

        ret_memberships = np.zeros(shape=(n_comparisons,n_samples)) 
        ret_nonmemberships =  np.zeros(shape=(n_comparisons,n_samples)) 
        ret_pro_misvs = np.full(shape=(n_comparisons,n_samples),fill_value=-1, dtype=int)
        ret_con_misvs = np.full(shape=(n_comparisons,n_samples),fill_value=-1, dtype=int)
       
         
        p2 = 0
        for i in range(n_classes):
            for j in range(i+1,n_classes):
                 
                part_class_i = weighted_vals[:,j-1,start[i]:end[i]]
                part_class_j = weighted_vals[:,i,  start[j]:end[j]]

                positives_class_i = np.greater(part_class_i,0) 
                positives_class_j = np.greater(part_class_j,0) 
                negatives_class_i = np.less(part_class_i,0) 
                negatives_class_j = np.less(part_class_j,0)

                influence_pro_i = np.sum(part_class_i, where=positives_class_i, axis=1) + np.sum(part_class_j, where=positives_class_j, axis=1)
                influence_pro_j = -(np.sum(part_class_i, where=negatives_class_i, axis=1) + np.sum(part_class_j, where=negatives_class_j, axis=1))

                max_influence_pro_i_misv_idx = np.argmax(part_class_i, axis=1) + start[i]
                max_influence_pro_j_misv_idx = np.argmax(-part_class_j, axis=1) + start[j]

                if b[p2]>0:
                    influence_pro_i+=b[p2]
                else:
                    influence_pro_j-=b[p2]
       
                ret_memberships[p2] = np.array([influence_pro_i])
                ret_nonmemberships[p2] = np.array([influence_pro_j])
                ret_pro_misvs[p2] = np.array([max_influence_pro_i_misv_idx])
                ret_con_misvs[p2] = np.array([max_influence_pro_j_misv_idx])

                p2 += 1

        if len(self.classes_) == 2:
            return ret_nonmemberships.transpose(), ret_memberships.transpose(), ret_con_misvs.transpose(), ret_pro_misvs.transpose()
        else:
            return ret_memberships.transpose(), ret_nonmemberships.transpose(), ret_pro_misvs.transpose(), ret_con_misvs.transpose()


    def decision_function_with_context(self, X):
        """ Evaluates the decision function for each sample in X.

        Parameters:
        X:  ndarray of shape (n_samples, n_features); or 
            ndarray of shape (n_features,). 

        Returns:
            A 4-tuple (memberships, nonmemberships, pro_MISVs, con_MISVs) consisting of 4 ndarrays of shape 
            (n_samples, n_classes * (n_classes-1) / 2). 

            Returns the decision function of the sample for each class in the model. Since decision_function_shape='ovo'
            is always used as multi-class strategy, df is an array of shape (n_samples, n_classes * (n_classes-1) / 2).

            N.B.: (memberships - nonmemberships) must be equal to decision_function(self, X)

        Notes:
            About decision_function_shape : {'ovo', 'ovr'}

            Whether to return a one-vs-rest ('ovr') decision function of shape
            (n_samples, n_classes) as all other classifiers, or the original
            one-vs-one ('ovo') decision function of libsvm which has shape
            (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
            ('ovo') is always used as multi-class strategy. The parameter is
            ignored for binary classification.

            See https://github.com/scikit-learn/scikit-learn/blob/23afd5d95c18915c55070cecaecf9f3030ae9bbb/sklearn/svm/_classes.py
                    
        """

        if isinstance(X,np.ndarray):
            if X.ndim == 1:
                return self.__decision_function_with_context_ndim(np.array([X]))
            else:
                return self.__decision_function_with_context_ndim(X)
        elif isinstance(X, list):
            return self.__decision_function_with_context_ndim(np.array(X))
        else:
            raise ValueError("X parameter must be an ndarray")

    
    def __binary_decision_function_with_context_1dim(self, X):
        """ Evaluates the binary decision function for sample X.

        Parameters:
        X: ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns:
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
    


    def __sort_votes(self, e):
        return e[1]

    def __sort_buoyancy(self, e):
        return e[1].buoyancy

    def __predict_with_context_by_voting_1dim(self, X):
        """Performs an augmented prediction of the top-K classes for sample X 
        
        Parameters:
        X:  ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns:
        topK: list of the top-K classes predicted for X.

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        cs = self.classes_
        k = self.k
        
        ifselements = self.__decision_function_with_context_1dim(X)

        if len(cs)==2:
            ifs_elem = ifselements[0]
            eta = max(1, ifs_elem.mu_hat.value + ifs_elem.nu_hat.value)
            ifs_elem = ifs_elem.normalize(eta)
            if ifs_elem.buoyancy > 0: 
                return [xPrediction(cs[1],ifs_elem)]
            else:
                return [xPrediction(cs[0],ifs_elem)]

        votes = [0 for c in range(len(cs))]
        evals = [xAIFSElement() for c in range(len(cs))]
        misvs = [xAIFSElement() for c in range(len(cs))]

        p = 0
        for i in range(len(cs)):
            for j in range(i+1,len(cs)):
                if ifselements[p].buoyancy > 0: # pro_i 
                    votes[i]+=1
                else: 
                    if ifselements[p].buoyancy < 0: # pro_j 
                        votes[j]+=1

                # pro_i 
                evals[i].mu_hat.value += ifselements[p].mu_hat.value
                evals[i].nu_hat.value += ifselements[p].nu_hat.value
                # pro_j
                evals[j].mu_hat.value += ifselements[p].nu_hat.value
                evals[j].nu_hat.value += ifselements[p].mu_hat.value    

                # Obtain MISVs per class
                if ifselements[p].mu_hat.value >= misvs[i].mu_hat.value:
                    misvs[i].mu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= misvs[i].nu_hat.value:
                    misvs[i].nu_hat = ifselements[p].nu_hat

                if ifselements[p].mu_hat.value >= misvs[j].nu_hat.value:
                    misvs[j].nu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= misvs[j].mu_hat.value:
                    misvs[j].mu_hat = ifselements[p].nu_hat 
                                  
                p+=1

        eta = 1
       
        for eval in evals:
            eta_eval = max(1, eval.mu_hat.value + eval.nu_hat.value)
            if eta_eval > eta:
                eta = eta_eval
    
        for i in range(len(cs)):
            evals[i].mu_hat.misv_idx = misvs[i].mu_hat.misv_idx
            evals[i].nu_hat.misv_idx = misvs[i].nu_hat.misv_idx
            evals[i] = evals[i].normalize(eta)
        
        indices_classes =  [c for c in range(len(cs))]  
        sorted_classes = list(zip(indices_classes, votes, evals))
        sorted_classes.sort(key=self.__sort_votes,reverse=True)

        # Return collection of xPrediction
        topK = [xPrediction(cs[t[0]],t[2]) for t in sorted_classes[0:k]]
       
        # For comparison with __predict_with_context_by_voting_ndim 
        # (N.B. #votes can be equal and sorting results can be different)
        # topK = [(xPrediction(cs[t[0]],t[2]),t[1] ) for t in sorted_classes[0:k]] 
        return topK

    def __predict_with_context_by_voting_ndim(self, X):
        """Performs an augmented prediction of the top-K classes for each sample in X. 
        
        Parameters:
        X:  ndarray of shape (n_samples, n_features). 

        Returns
        topK: ndarray of shape (n_samples, k) consisting of the top-K classes predicted for each sample in X. 

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")

        cs = self.classes_
        n_classes = len(cs)
        k = self.k
        
        memberships, nonmemberships, membership_misvs, nonmembership_misvs = self.decision_function_with_context(X)

        n_samples = memberships.shape[0]

        if n_classes==2:
            buoyancy = memberships - nonmemberships
            eta = np.max(memberships + nonmemberships, axis=1)
            eta = np.where(eta>1.0, eta, 1.0)
            ret = np.full(shape=(n_samples,1),fill_value=None)
            for i in range(n_samples):
                membershipAAD = xAAD(memberships[i][0], membership_misvs[i][0])
                nonmembershipAAD = xAAD(nonmemberships[i][0], nonmembership_misvs[i][0])
                ifsElem = xAIFSElement(membershipAAD, nonmembershipAAD)
                ifsElem = ifsElem.normalize(eta[i])
                if buoyancy[i][0] > 0: 
                   prediction = xPrediction(cs[1], ifsElem)
                else:
                   prediction = xPrediction(cs[0], ifsElem)

                ret[i] = prediction
            return ret
    

        votes_per_class = np.zeros(shape=(n_samples,n_classes),dtype=int)
        memberships_per_class = np.zeros(shape=(n_samples,n_classes))
        nonmemberships_per_class = np.zeros(shape=(n_samples,n_classes))
        membershipMISVs_per_class = np.full(shape=(n_samples,n_classes),fill_value=-1, dtype=int)
        nonmembershipMISVs_per_class = np.full(shape=(n_samples,n_classes),fill_value=-1, dtype=int)
        max_membership_per_class = np.zeros(shape=(n_samples,n_classes))
        max_nonmembership_per_class = np.zeros(shape=(n_samples,n_classes))

        buoyancy = memberships - nonmemberships
        pro_i = np.greater(buoyancy,0)
        pro_j = np.less(buoyancy,0)


        p = 0
        for i in range(len(cs)):
            for j in range(i+1,len(cs)):
                # counting
                votes_per_class[:,i] += np.where(pro_i[:,p],1,0)
                votes_per_class[:,j] += np.where(pro_j[:,p],1,0)
       
                # # pro_i 
                memberships_per_class[:,i] += memberships[:,p]
                nonmemberships_per_class[:,i] += nonmemberships[:,p]

                # # pro_j
                memberships_per_class[:,j] += nonmemberships[:,p]
                nonmemberships_per_class[:,j] += memberships[:,p]

                # # Obtain MISVs per class
                cond = memberships[:, p] >=  max_membership_per_class[:,i]
                max_membership_per_class[:,i] = np.where(cond, memberships[:, p], max_membership_per_class[:,i] )
                membershipMISVs_per_class[:,i] = np.where(cond, membership_misvs[:, p], membershipMISVs_per_class[:,i] )
                
                cond = nonmemberships[:, p] >= max_nonmembership_per_class[:,i]
                max_nonmembership_per_class[:,i] = np.where(cond, nonmemberships[:, p], max_nonmembership_per_class[:,i] )
                nonmembershipMISVs_per_class[:,i] = np.where(cond, nonmembership_misvs[:, p], nonmembershipMISVs_per_class[:,i])

                cond = memberships[:, p] >= max_nonmembership_per_class[:,j]
                max_nonmembership_per_class[:,j] = np.where(cond, memberships[:, p],  max_nonmembership_per_class[:,j] )
                nonmembershipMISVs_per_class[:,j] = np.where(cond, membership_misvs[:, p], nonmembershipMISVs_per_class[:,j], )


                cond = nonmemberships[:, p] >= max_membership_per_class[:,j]
                max_membership_per_class[:,j] = np.where(cond, nonmemberships[:, p], max_membership_per_class[:,j] )
                membershipMISVs_per_class[:,j] = np.where(cond, nonmembership_misvs[:, p], membershipMISVs_per_class[:,j])
                                  
                p+=1


        eta = np.max(memberships_per_class + nonmemberships_per_class, axis=1)
        eta = np.where(eta>1.0, eta, 1.0)


        # best_classes = np.argmax(votes_per_class,axis=1)
        sorted_idx = np.argsort(votes_per_class, axis=1)
        sorted_idx_desc = np.flip(sorted_idx, axis=1)
        k_best_idx = sorted_idx_desc[:,:k]

        ret = np.full(shape=(n_samples,k),fill_value=None)
        for i in range(n_samples):
            for ik, j in enumerate(k_best_idx[i]):
                membershipAAD = xAAD(memberships_per_class[i,j], membershipMISVs_per_class[i,j])
                nonmembershipAAD = xAAD(nonmemberships_per_class[i,j], nonmembershipMISVs_per_class[i,j])
                ifsElem = xAIFSElement(membershipAAD, nonmembershipAAD)
                ifsElem = ifsElem.normalize(eta[i])
                prediction = xPrediction(cs[j], ifsElem)
                ret[i,ik] = prediction

        return ret

   

    def predict_with_context_by_voting(self, X):
        """Performs an augmented prediction of the top-K classes for each sample in X.

        
        Parameters:
        X: ndarray of shape (n_samples, n_features); or
           ndarray of shape (n_features,) consisting of n features identified for sample X. 
        
        Returns:
        topK: ndarray of shape (n_samples, k) consisting of the top-K classes predicted for each sample in X; or
              list of the top-K classes predicted for X.

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        if isinstance(X,np.ndarray):
            if X.ndim == 1:
                return self.__predict_with_context_by_voting_1dim(X)
            else:
                return self.__predict_with_context_by_voting_ndim(X)
        elif isinstance(X, list):
            return self.__predict_with_context_by_voting_ndim(np.array(X))
        else:
            raise ValueError("X parameter must be an ndarray")



    def __evaluate_all_memberships_1dim(self, X):
        """Performs augmented evaluation of the proposition 'X IS A' for each class A learned during the training process.
        
        Parameters:
        X:  ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns:
        arr : array of n xIFSElements representing the augmented evaluations.
        """

        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        cs = self.classes_
        
        ifselements = self.__decision_function_with_context_1dim(X)
      
        if len(cs)==2:
            eta = max(1,ifselements[0].mu_hat.value + ifselements[0].nu_hat.value)
            ifs_elem = ifselements[0]
            ifs_elem = ifs_elem.normalize(eta)
            opposite = xAIFSElement(ifs_elem.nu_hat, ifs_elem.mu_hat)
            return [ifs_elem, opposite]

        evals = [xAIFSElement() for c in range(len(cs))]
        misvs = [xAIFSElement() for c in range(len(cs))]

        p = 0
        for i in range(len(cs)):
            for j in range(i+1,len(cs)):
                # pro_i 
                evals[i].mu_hat.value += ifselements[p].mu_hat.value
                evals[i].nu_hat.value += ifselements[p].nu_hat.value
                # pro_j
                evals[j].mu_hat.value += ifselements[p].nu_hat.value
                evals[j].nu_hat.value += ifselements[p].mu_hat.value    

                # MISVs per class
                if ifselements[p].mu_hat.value >= misvs[i].mu_hat.value:
                    misvs[i].mu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= misvs[i].nu_hat.value:
                    misvs[i].nu_hat = ifselements[p].nu_hat

                if ifselements[p].mu_hat.value >= misvs[j].nu_hat.value:
                    misvs[j].nu_hat = ifselements[p].mu_hat
                if ifselements[p].nu_hat.value >= misvs[j].mu_hat.value:
                    misvs[j].mu_hat = ifselements[p].nu_hat  

                # print("MISV_{0}: ({1},{2}), MISV_{3}: ({4},{5}) p: {6}".format(
                #     i, misvs[i].mu_hat.misv_idx , misvs[i].nu_hat.misv_idx,
                #     j, misvs[j].mu_hat.misv_idx , misvs[j].nu_hat.misv_idx, p))

                p+=1

                

        eta = 1
        for eval in evals:
            eta_eval = max(1, eval.mu_hat.value + eval.nu_hat.value)
            if eta_eval > eta:
                eta = eta_eval
    
        for i in range(len(cs)):
            evals[i].mu_hat.misv_idx = misvs[i].mu_hat.misv_idx
            evals[i].nu_hat.misv_idx = misvs[i].nu_hat.misv_idx
            evals[i] = evals[i].normalize(eta)

        return evals

    def __evaluate_all_memberships_ndim(self, X):
        """Performs augmented evaluation of the proposition 'X IS A' for each class A learned during the training process.
        
        Parameters:
        X:  ndarray of shape (n_samples, n_features). 

        Returns:
        arr : ndarray of shape (n_samples, n_classes) consisting of the augmented evaluations.
        """
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")

        cs = self.classes_
        n_classes = len(cs)
        
        memberships, nonmemberships, membership_misvs, nonmembership_misvs = self.decision_function_with_context(X)

        n_samples = memberships.shape[0]
        
        memberships_per_class = np.zeros(shape=(n_samples,n_classes))
        nonmemberships_per_class = np.zeros(shape=(n_samples,n_classes))
        membershipMISVs_per_class = np.full(shape=(n_samples,n_classes),fill_value=-1, dtype=int)
        nonmembershipMISVs_per_class = np.full(shape=(n_samples,n_classes),fill_value=-1, dtype=int)
        max_membership_per_class = np.zeros(shape=(n_samples,n_classes))
        max_nonmembership_per_class = np.zeros(shape=(n_samples,n_classes))

        p = 0
        for i in range(len(cs)):
            for j in range(i+1,len(cs)):
                # pro_i 
                memberships_per_class[:,i] += memberships[:,p]
                nonmemberships_per_class[:,i] += nonmemberships[:,p]

                # pro_j
                memberships_per_class[:,j] += nonmemberships[:,p]
                nonmemberships_per_class[:,j] += memberships[:,p]

                # MISVs per class
                cond = memberships[:, p] >=  max_membership_per_class[:,i]
                max_membership_per_class[:,i] = np.where(cond, memberships[:, p], max_membership_per_class[:,i] )
                membershipMISVs_per_class[:,i] = np.where(cond, membership_misvs[:, p], membershipMISVs_per_class[:,i] )
               
                cond = nonmemberships[:, p] >= max_nonmembership_per_class[:,i]
                max_nonmembership_per_class[:,i] = np.where(cond, nonmemberships[:, p], max_nonmembership_per_class[:,i] )
                nonmembershipMISVs_per_class[:,i] = np.where(cond, nonmembership_misvs[:, p], nonmembershipMISVs_per_class[:,i])

                cond = memberships[:, p] >= max_nonmembership_per_class[:,j]
                max_nonmembership_per_class[:,j] = np.where(cond, memberships[:, p],  max_nonmembership_per_class[:,j] )
                nonmembershipMISVs_per_class[:,j] = np.where(cond, membership_misvs[:, p], nonmembershipMISVs_per_class[:,j], )

                cond = nonmemberships[:, p] >= max_membership_per_class[:,j]
                max_membership_per_class[:,j] = np.where(cond, nonmemberships[:, p], max_membership_per_class[:,j] )
                membershipMISVs_per_class[:,j] = np.where(cond, nonmembership_misvs[:, p], membershipMISVs_per_class[:,j])
              
                p+=1


        eta = np.max(memberships_per_class + nonmemberships_per_class, axis=1)
        eta = np.where(eta>1.0, eta, 1.0)

        ret = np.full(shape=(n_samples,n_classes),fill_value=None)
        for i in range(n_samples):
            for j in range(n_classes):
                membershipAAD = xAAD(memberships_per_class[i,j], membershipMISVs_per_class[i,j])
                nonmembershipAAD = xAAD(nonmemberships_per_class[i,j], nonmembershipMISVs_per_class[i,j])
                ifsElem = xAIFSElement(membershipAAD, nonmembershipAAD)
                ifsElem = ifsElem.normalize(eta[i])
                ret[i,j] = ifsElem

        return ret



    def evaluate_all_memberships(self, X):
        """Performs augmented evaluation of the proposition 'X IS A' for each class A learned during the training process
        for each sample in X.
        
        Parameters:
        X:  ndarray of shape (n_samples, n_features); or
            ndarray of shape (n_features,). 

        Returns:
        arr : ndarray of shape (n_samples, n_classes) consisting of the augmented evaluations.
        """
        if isinstance(X,np.ndarray):
            if X.ndim == 1:
                return self.__evaluate_all_memberships_1dim(X)
            else:
                return self.__evaluate_all_memberships_ndim(X)
        elif isinstance(X, list):
            return self.__evaluate_all_memberships_ndim(np.array(X))
        else:
            raise ValueError("X parameter must be an ndarray")



    def  is_member_of(self, X, class_idx):
        """Performs an augmented evaluation of 'X IS A', where A is the class referenced by class_idx
        for each sample in X.
        
        Parameters
        X:  ndarray of shape (n_samples, n_features); or
            ndarray of shape (n_features,)

        class_idx: class index

        Returns
        evals : ndarray of shape (n_samples) consisting of the augmented evaluations.
        """

        evals = self.evaluate_all_memberships(X)

        if isinstance(X,np.ndarray):
            if X.ndim == 1:
                return evals[class_idx]
            else:
                return evals[:, class_idx]
        elif isinstance(X, list):
            return evals[class_idx]
        else:
            raise ValueError("X parameter must be an ndarray")
        


    
    def __predict_with_context_1dim(self, X):
        """Performs an augmented prediction of the top-K classes for X 
        
        Parameters
        X :  ndarray of shape (n_features,) consisting of n features identified for X. 

        Returns
        topK : list of the top-K classes predicted for X.

        """
        cs = self.classes_
        k = self.k
        evals = self.evaluate_all_memberships(X)

        if len(cs)==2:
            if evals[0].buoyancy > 0: 
                return [xPrediction(cs[1],evals[0])]
            else:
                return [xPrediction(cs[0],evals[0])]
        
        indices_classes =  [c for c in range(len(cs))]  
        sorted_classes = list(zip(indices_classes, evals))
        sorted_classes.sort(key=self.__sort_buoyancy,reverse=True)

        # Return collection of xPrediction
        topK = [xPrediction(cs[t[0]],t[1]) for t in sorted_classes[0:k]]
        return topK


    def __predict_with_context_ndim(self, X):
        """Performs an augmented prediction of the top-K classes for each sample in X 
        
        Parameters
        X :  ndarray of shape (n_samples, n_features). 

        Returns
        topK : ndarray of shape (n_samples, k) consisting of the top-K classes predicted for X.

        """
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")

        cs = self.classes_
        n_classes = len(cs)
        k = self.k
        n_samples = X.shape[0]

        evals = self.evaluate_all_memberships(X)

        if len(cs)==2:
            ret = np.full(shape=(n_samples,1),fill_value=None)
            for i in range(n_samples):
                evals_per_sample = evals[i]
                if evals_per_sample[0].buoyancy > 0: 
                    ret[i,:] = np.array([xPrediction(cs[1],evals_per_sample[0])])
                else:
                    ret[i,:] = np.array([xPrediction(cs[0],evals_per_sample[0])])
            return ret
        

        indices_classes =  [c for c in range(len(cs))]  
        ret = np.full(shape=(n_samples,k),fill_value=None)
        for i in range(n_samples):
            evals_per_sample = evals[i]
            sorted_classes = list(zip(indices_classes, evals_per_sample))
            sorted_classes.sort(key=self.__sort_buoyancy,reverse=True)
            topK = [xPrediction(cs[t[0]],t[1]) for t in sorted_classes[0:k]]
            ret[i,:] = np.array(topK)

        return ret

    def predict_with_context(self, X):
        """Performs an augmented prediction of the top-K classes for each sample in X 
        
        Parameters
        X :  ndarray of shape (n_samples, n_features), or
             ndarray of shape (n_features)

        Returns
        topK : ndarray of shape (n_samples, k) consisting of the top-K classes predicted for X.

        """
        if isinstance(X,np.ndarray):
            if X.ndim == 1:
                return self.__predict_with_context_1dim(X)
            else:
                return self.__predict_with_context_ndim(X)
        elif isinstance(X, list):
            return self.__predict_with_context_ndim(np.array(X))
        else:
            raise ValueError("X parameter must be an ndarray")