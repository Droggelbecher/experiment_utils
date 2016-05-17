
import numpy as np
import metrics

class Features:
    def __init__(self, *args):
        self.features = args

        p = 0
        for f in self.features:
            f.start = p
            f.end = p + len(f.keys)
            setattr(self, f.name, f)
            p += len(f.keys)
        self._len = p

    def __contains__(self, fname):
        return hasattr(self, fname) and type(getattr(self, fname)) is Feature

    def distance(self, a, b, features=None, ignore_features=(), weights='default'):
        # sklearn calls this for testing with random integers (why the heck??!)
        if len(a) != self._len:
            return float('nan')

        if features is None:
            features = set([f.name for f in self.features])
        features = set(features) - set(ignore_features)

        w = 1.0 / len(features)

        d = 0.0
        for fname in features:
            f = getattr(self, fname)
            if weights == 'default':
                w = f.weight
            d += f.distance(a, b) * w
        return d

    def assemble(self, **kws):
        r = np.zeros(self._len)
        idx = 0
        rest_idx = 0
        rest = np.zeros(self._len)
        if '_rest' in kws and kws['_rest'] is not None:
            rest = kws['_rest']

        while idx < self._len:
            for k, v in kws.items():
                if k == '_rest': continue
                f = getattr(self, k)
                if f.start == idx:
                    r[f.start:f.end] = v
                    idx = f.end + 1
                    break
            else:
                r[idx] = rest[rest_idx]
                idx += 1
                rest_idx += 1

        return r

    def all_except(self, featname, a):
        f = getattr(self, featname)
        return np.hstack((a[:f.start], a[f.end:]))


class Feature:
    def __init__(self, name, keys, type_='set', **kws):
        self.name = name
        self.keys = keys

        self.distance = getattr(self, type_ + '_distance', None)
        self.mean = getattr(self, type_ + '_mean', None)

        self.__dict__.update(kws)

    def __call__(self, a):
        return a[self.start:self.end]

    def __len__(self):
        return len(self.keys)

    #
    # Categorial data, "one hot" encoding
    # One value is assumed to be == 1.0 the rest to be == 0.0
    # Warning: You will likely destroy this property when doing things like
    # averaging, PCA, etc... across data rows
    #

    def cat_distance(self, a, b):
        pa = np.where(self(a) == 1)[0][0]
        pb = np.where(self(b) == 1)[0][0]
        return pa - pb

    def cat_wrap_distance(self, a, b):
        d = self.dist_cat(a, b)
        if d < 0:
            return d + len(self)
        return d

    def cat_mean(self, a):
        """
        Feature is represented as a vector of bools,
        keys are associated values.
        Average over the values that are "true".
        """
        return np.average(self.__call__(a) * self.keys)

    #
    # Set, vector of 1s and 0s,
    # Like categorial but any number of 0s/1s allowed
    #

    def set_distance(self, a, b):
        return metrics.jaccard(self(a), self(b))

    #
    # Histogram
    # Like set but allow arbitrary values
    #

    def hist_distance(self, a, b):
        return metrics.chi_squared(self(a), self(b))

    #
    # Histogram
    # Distance corresponds to difference of averages
    # (and thus focusses on the X-Axis rather than Y)
    #

    def chist_distance(self, a, b):
        return (np.average(self(a) * self.keys) - np.average(self(b) * self.keys)) / max(self.keys)

    def chist_wrap_distance(self, a, b):
        d = self.chist_distance(a, b)
        if d < 0:
            return d + 1.0
        return d

    #
    # Geo location (latitude, longitude)
    #

    def geo_distance(self, a, b):
        return metrics.geo(self(a), self(b))



