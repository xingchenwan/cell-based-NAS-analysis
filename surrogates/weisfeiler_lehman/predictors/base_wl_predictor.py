from surrogates.base_predictor import BasePredictor


class BaseWLPredictor(BasePredictor):

    def __init__(self, h: int = 1):
        """
        The base class for predictors based on WL feature extractor.
        :param h:  int. Number of Weisfeiler-Lehman Iterations
        """
        super().__init__()
        self.h = h
        # Save history for the input and targets for the current predictor class
        self.X, self.y = None, None
        # the lower bound and upper bound for feature vector X and the mean and std of target vector Y
        # will be initialised when the GPWL model is fitted to
        #   some data.
        self.lb, self.ub = None, None
        self.ymean, self.ystd = None, None
