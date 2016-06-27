
import numpy as np

def sparsly(a):
    s = 'ar[\n  shape: {}\n'.format(a.shape)

    idxs = np.where(a != 0)
    for t in zip(*idxs):
        s += '  {} = {}\n'.format(t, a[t])
    return s + ']'

