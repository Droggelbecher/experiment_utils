
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

    def __len__(self):
        return self._len

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
            d += f.distance(f(a), f(b)) * w
        return d

    def assemble(self, **kws):
        """
        kws: feature name => value (=ndarray of correct size).
            '_rest' will be used to fill up all non-addressed feature columns

        return: ndarray of full row length with inserted features
        """
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
                    if v is not None:
                        r[f.start:f.end] = v
                    idx = f.end + 1
                    break
            else:
                r[idx] = rest[rest_idx]
                idx += 1
                rest_idx += 1

        return r

    def get_feature(self, fname):
        assert isinstance(fname, str)
        return getattr(self, fname)

    def get_features(self, names):
        for f in self.features:
            if f.name in names:
                yield f

    def get_keys(self, featurenames):
        r = []
        features = list(self.get_features(featurenames))
        for f in features:
            if len(f.keys) == 1:
                r.append(f.name)
            else:
                for k in f.keys:
                    r.append('{}.{}'.format(f.name, k))
        return r

    def get_names(self):
        return set(f.name for f in self.features)

    def extract(self, a, fnames):
        """
        a: full data row to extract features from
        features: iterable over feature names to extract
        return: ndarray of requested features in row order
        """
        features = list(self.get_features(fnames))
        l = sum(len(f) for f in features)
        r = np.zeros(shape = a.shape[:-1] + (l,))
        i = 0
        for f in features:
            r[..., i:i + len(f)] = f(a)
            i += len(f)

        return r

    def all_except(self, featname, a):
        f = getattr(self, featname)
        return np.hstack((a[:f.start], a[f.end:]))

class Feature:
    keys = ('value',)

    def __init__(self, name):
        self.name = name

    def __call__(self, a):
        return a[...,self.start:self.end]

    def encode(self, a, v):
        a[...,self.start:self.end] = v

    def decode(self, a):
        return a[...,self.start:self.end]

    def __len__(self):
        return len(self.keys)

class PlainFeature(Feature):
    def __init__(self, name, range_ = None):
        Feature.__init__(self, name)
        self._range = range_
        if self._range is not None:
            self._delta = float(self._range[1] - self._range[0])

    def distance(self, a, b):
        d = a - b
        if self._range is not None:
            return d / self._delta
        return d

class OneHotFeature(Feature):
    def __init__(self, name, keys, wrap = False):
        Feature.__init__(self, name)
        self.keys = keys
        self._wrap = wrap

    def distance(self, a, b):
        pa = np.where(a == 1)[0][0]
        pb = np.where(b == 1)[0][0]
        d = pa - pb
        if self._wrap and d < 0:
            return d + len(self)
        return d

    def encode(self, a, v):
        v2 = np.zeros(len(self))
        v2[v] = 1
        Feature.encode(self, a, v2)

    def decode(self, a):
        a = Feature.decode(self, a)
        return np.average(a * self.keys)

class BitSetFeature(Feature):
    def __init__(self, name, keys):
        Feature.__init__(self, name)
        self.keys = keys

    def distance(self, a, b):
        return metrics.jaccard(a, b)

class HistogramFeature(Feature):
    def __init__(self, name, metric = 'chisquared', wrap = False):
        Feature.__init__(self, name)
        self._metric = metric
        self._wrap = wrap

    def distance(self, a, b):
        if self.metric == 'chisquared':
            assert not self._wrap
            return metrics.chi_squared(a, b)
        else:
            d = (np.average(a * self.keys) - np.average(b * self.keys)) / max(self.keys)
            if self._wrap and d < 0:
                return d + 1.0
            return d

class GeoFeature(Feature):
    keys = ('lat', 'lng')
    def distance(self, a, b):
        return metrics.geo(a, b)


