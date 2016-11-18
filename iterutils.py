
import math
import itertools

def chunks(list_, n):
    """
    Split list-like list_ into n equally sized chunks (the last one may be
    smaller).

    >>> l = list(range(10, 75))
    >>> list( chunks(l, 7) )
    [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59], [60, 61, 62, 63, 64, 65, 66, 67, 68, 69], [70, 71, 72, 73, 74]]
    """

    sz = int(math.ceil(len(list_) / float(n)))
    for i in range(0, len(list_), sz):
        yield list_[i:i + sz]


def unique(list_):

    from collections import OrderedDict

    return list(OrderedDict.fromkeys(list_))

def flatten(list_):
    """
    >>> l = [ [ [ 1, 2 ], [ 3, [ 4, 5, 6 ] ], [], [ [ [ ] ] ], 7 ], 8 ]
    >>> flatten(l)
    [1, 2, 3, 4, 5, 6, 7, 8]
    """
    r = []
    for item in list_:
        if type(item) in (tuple, list):
            r.extend(flatten(item))
        else:
            r.append(item)
    return r

def repeat_to(l, n):
    """
    >>> repeat_to([1, 2, 3], 10)
    (1, 2, 3, 1, 2, 3, 1, 2, 3, 1)
    """
    return tuple(itertools.islice(itertools.cycle(l), n))

def neighbors(l):
    """
    >>> neighbors(list(range(10)))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    """
    return zip(l[:-1], l[1:])

def cyc_neighbors(l):
    """
    >>> cyc_neighbors(list(range(10)))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]

    >>> import numpy as np
    >>> a = np.array(list(range(10)))
    >>> cyc_neighbors(a)
    array([[0, 1],
           [1, 2],
           [2, 3],
           [3, 4],
           [4, 5],
           [5, 6],
           [6, 7],
           [7, 8],
           [8, 9],
           [9, 0]])
    >>> aa = np.array((a, a)).T
    >>> cyc_neighbors(aa)
    array([[[0, 0],
            [1, 1]],
    <BLANKLINE>
           [[1, 1],
            [2, 2]],
    <BLANKLINE>
           [[2, 2],
            [3, 3]],
    <BLANKLINE>
           [[3, 3],
            [4, 4]],
    <BLANKLINE>
           [[4, 4],
            [5, 5]],
    <BLANKLINE>
           [[5, 5],
            [6, 6]],
    <BLANKLINE>
           [[6, 6],
            [7, 7]],
    <BLANKLINE>
           [[7, 7],
            [8, 8]],
    <BLANKLINE>
           [[8, 8],
            [9, 9]],
    <BLANKLINE>
           [[9, 9],
            [0, 0]]])

    """
    if isinstance(l, list):
        return zip(l, l[1:] + l[:1])
    else:
        import numpy as np
        return np.array(zip(l, np.roll(l, -1, axis = 0)))

class CV:

    """
    >>> n = 78
    >>> parts = 10
    >>> cv = CV(n, parts)
    >>> l = list(cv)
    >>> len(l) == parts
    True
    >>> l[0][0] == 0
    True
    >>> l[-1][1] == n
    True
    >>> l
    [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48), (48, 56), (56, 64), (64, 72), (72, 78)]
    """

    def __init__(self, n, parts):
        self.n = n
        self.parts = parts
        self._sz = int(math.ceil(n / float(parts)))
        assert self._sz >= 1

    def __iter__(self):

        for i in range(0, self.n, self._sz):
            yield i, min(i + self._sz, self.n)

