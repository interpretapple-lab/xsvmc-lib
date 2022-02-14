

class AAD():
    """ Augmented Appraisal Degree 
        
        This class is an implementation of an augmented appraisal degree (AAD) [1], which is a generalization of
        a membership grade [2]. An AAD denotes to which level and hints why a (membership) criterion is fulfilled [2].

        References:

        [1] M. Loor, G and De Tré, On the need for augmented appraisal degrees to handle experience-based evaluations,
            Applied Soft Computing, Volume 54, 2017, Pages 284-295, ISSN 1568-4946,
            https://doi.org/10.1016/j.asoc.2017.01.009. 

        [2] L. Zadeh, Fuzzy sets, Inf. Control 8 (3) (1965) 338–353, http://dx.doi.org/10.1016/S0019-9958(65)90241-X.

    """
    def __init__(self, level=0, reason=None):
        self.level = level #This is the level
        self.reason = reason #This is the reason


class AIFSElement():
    """ Augmented Intuitinistc Fuzzy Set Element

        This class is an implementation of an augmented intuitinistic fuzzy set element (AIFSElement) [1], which is a generalization of
        an intuitinistic fuzzy set element [2]. 

    References:
        [1] M. Loor and G. De Tré, On the need for augmented appraisal degrees to handle experience-based evaluations,
            Applied Soft Computing, Volume 54, 2017, Pages 284-295, ISSN 1568-4946,
            https://doi.org/10.1016/j.asoc.2017.01.009.
            
        [2] K. Atanassov, Intuitionistic fuzzy sets, Fuzzy Sets and Systems, Volume 20, Issue 1, 1986,
            Pages 87-96, ISSN 0165-0114, https://doi.org/10.1016/S0165-0114(86)80034-3.

    """
    def __init__(self, object=None, membership=AAD(), nonmembership=AAD()):
        self.object = object
        self.membership = membership
        self.nonmembership = nonmembership
    

    @property
    def buoyancy(self):
        """
        This property implements the buoyancy of an IFSElement as proposed in [1].

        Reference:
            [1] Loor, M., Tapia-Rosero, A. and De Tré, G. Usability of Concordance Indices in FAST-GDM Problems. 
                In Proceedings of the 10th International Joint Conference on Computational Intelligence (IJCCI 2018), 
                pages 67-78 ISBN: 978-989-758-327-8. https://doi.org/10.5220/0006956500670078.
        """
        return self.membership.level - self.nonmembership.level
    
    @property
    def hesitation(self):
        """
        This property implements the hesitation margin of an IFSElement [1].

        Reference:
            [1] K. Atanassov, Intuitionistic fuzzy sets, Fuzzy Sets and Systems, Volume 20, Issue 1, 1986,
            Pages 87-96, ISSN 0165-0114, https://doi.org/10.1016/S0165-0114(86)80034-3.
        return self.membership.level - self.nonmembership.level
        """
        return 1.0 - self.membership.level - self.nonmembership.level
        

class xAAD(AAD):
    """ Augmented Appraisal Degree with the (index of the) most influential support vector (MISV)
    """
    def __init__(self, value = 0.0, idx = -1):
        self.level = value
        self.reason = idx

    @property
    def value(self):
        return self.level
    
    @value.setter
    def value(self, value):
        self.level = value

    @property
    def misv_idx(self):
        return self.reason
    
    @misv_idx.setter
    def misv_idx(self, idx):
        self.reason = idx



class xAIFSElement(AIFSElement):
    """ IFSElement with xAADs 
    """
    def __init__(self, mu_hat = None, nu_hat = None):
        if(isinstance(mu_hat, (xAAD))):
            self.mu_hat = mu_hat
        else:
            self.mu_hat = xAAD()

        if(isinstance(nu_hat, (xAAD))):
            self.nu_hat = nu_hat
        else:  
            self.nu_hat = xAAD()


    @property
    def mu_hat(self):
        return self.membership

    @mu_hat.setter
    def mu_hat(self, membershipAAD):
        self.membership = membershipAAD
    
    @property
    def nu_hat(self):
        return self.nonmembership

    @nu_hat.setter
    def nu_hat(self, nonmembershipAAD):
        self.nonmembership = nonmembershipAAD
    
    
    def normalize(self, eta):
        mu = xAAD(self.mu_hat.level/eta, self.mu_hat.misv_idx)
        nu = xAAD(self.nu_hat.level/eta, self.nu_hat.misv_idx)
        return xAIFSElement(mu, nu)

  

class xPrediction:
    """ Augmented Prediction with the Most Influential Support Vectors
    """
    def __init__(self, class_name, eval):
        if(not isinstance(eval, (xAIFSElement))):
            raise ValueError("class_idx parameter must be an xIFSElement")
        self.class_name = class_name
        self.eval = eval
        