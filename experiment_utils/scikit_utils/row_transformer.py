
import numpy as np

class RowTransformer:
    def __init__(self, rowfunc):
        self.rowfunc = rowfunc

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        r = []
        for row in X:
            r.append(self.rowfunc(row))
        return np.array(r)

