
import numpy as np

class Features:
    def __init__(self, *args):
        self.features = args

        p = 0
        for f in self.features:
            f.start = p
            f.end = p + len(f.keys)
            setattr(self, f.name, f)
            p += len(f.keys)

class Feature:
    def __init__(self, name, keys, **kws):
        self.name = name
        self.keys = keys
        self.__dict__.update(kws)

    def __call__(self, a):
        return a[self.start:self.end]

    def __len__(self):
        return len(self.keys)

    def b_mean(self, a):
        """
        Fetaure is represented as a vector of bools,
        keys are associated values.
        Average over the values that are "true".
        """
        return np.average(self.__call__(a) * self.keys)

