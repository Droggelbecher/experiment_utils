
import math

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
    r = []
    for item in list_:
        if type(item) in (tuple, list):
            r.extend(flatten(item))
        else:
            r.append(item)
    return r

def repeat_to(l, n):
    return tuple(itertools.islice(itertools.cycle(l), n))


class CV:
    def __init__(self, n, parts):
        self.n = n
        self.parts = parts
        self._sz = int(math.ceil(n / float(parts)))

    def __iter__(self):

        for i in range(0, self.n, self._sz):
            yield i, min(i + self._sz, self.n)

