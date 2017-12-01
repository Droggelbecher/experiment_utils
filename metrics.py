
import math
import numpy as np

def jaccard(r1, r2):
    rr1 = r1 != 0
    rr2 = r2 != 0
    union = np.count_nonzero(rr1 | rr2)
    if union == 0:
        return 1.0
    return 1.0 - np.count_nonzero(rr1 & rr2) / float(union)

def fuzzy_jaccard(r1, r2):
    union = np.sum(r1 + r2)
    if union == 0:
        return 1.0
    return 1.0 - float(np.sum(a * b)) / float(union)

def chi_squared(r1, r2):
    rr1 = r1[:]
    rr2 = r2[:]
    eps = 0.0001
    rr1[r1 + r2 == 0] = eps
    rr1[r1 + r2 == 0] = eps
    return np.sum((rr1 - rr2) ** 2 / (rr1 + rr2))


