
import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Binner(BaseEstimator, TransformerMixin):

    def __init__(self, index, min_, max_, bin_width):
        self.index = index
        self.min_ = min_
        self.max_ = max_
        self.bin_width = bin_width

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xpre = X[..., :self.index]
        Xold = X[..., self.index]
        Xpost = X[..., self.index + 1:]

        bins = int(math.ceil((self.max_ - self.min_) / self.bin_width))

        Xnew = np.zeros(shape = (Xold.shape[0], bins))
        for i, x in enumerate(Xold):
            pos = int(bins * x / (self.max_ - self.min_))
            assert 0 <= pos < bins
            Xnew[i, pos] = 1

        return np.hstack((Xpre, Xnew, Xpost))

