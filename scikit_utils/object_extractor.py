
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ObjectExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, properties, **kws):
        self.properties = properties
        super(ObjectExtractor, self).__init__(**kws)

    def fit(self, X, y = None):
        return self

    def transform(self, objs):
        return np.array([
            [getattr(obj, x) for x in self.properties]
            for obj in objs
            ])



