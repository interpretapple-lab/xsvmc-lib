class xAAD:
    """ Augmented Appraisal Degree with Most Influential Support Vector
    """
    def __init__(self, value = 0.0, idx = -1):
        self.value = value
        self.misv_idx = idx

class xAIFSElement:
    """ IFSElement with xAADs 
    """
    def __init__(self):
        self.mu_hat = xAAD()
        self.nu_hat = xAAD()

    @property
    def buoyancy(self):
        return self.mu_hat.value - self.nu_hat.value

class xPrediction:
    """ Augmented Prediction with Most Influential Support Vector
    """
    def __init__(self, class_name, eval):
        if(not isinstance(eval, xAIFSElement)):
            raise ValueError("class_idx parameter must be an xIFSElement")
        self.class_name = class_name
        self.eval = eval
        