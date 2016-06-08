
from sklearn.base import BaseEstimator, TransformerMixin

class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, f):
        self.f = f

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        r = X[self.f(X), :]
        return r

