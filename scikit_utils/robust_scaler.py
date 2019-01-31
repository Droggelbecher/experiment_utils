
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RobustScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        self.p25_, self.median_, self.p75_ = np.percentile(X, [25, 50, 75], axis = 0)
        return self

    def transform(self, X):
        divisors = self.p75_ - self.p25_
        divisors[divisors == 0.0] = 1.0
        return (X - self.median_) / divisors

