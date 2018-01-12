
import math
import itertools
from typing import Sequence, Any, Tuple, Iterable, List

def flatten(list_: Iterable[Any]) -> List[Any]:
    """
    >>> l = [ [ [ 1, 2 ], [ 3, [ 4, 5, 6 ] ], [], [ [ [ ] ] ], 7 ], 8 ]
    >>> flatten(l)
    [1, 2, 3, 4, 5, 6, 7, 8]
    """
    r: List[Sequence[Any]] = []
    for item in list_:
        if type(item) in (tuple, list):
            r.extend(flatten(item))
        else:
            r.append(item)
    return r

def repeat_to(l: Iterable[Any], n: int) -> Tuple[Any, ...]:
    """
    >>> repeat_to([1, 2, 3], 10)
    (1, 2, 3, 1, 2, 3, 1, 2, 3, 1)
    """
    return tuple(itertools.islice(itertools.cycle(l), n))

def neighbors(l: Sequence[Any]) -> Iterable[Tuple[Any, Any]]:
    """
    >>> list( neighbors(list(range(10))) )
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    """
    return zip(l[:-1], l[1:])

def cyc_neighbors(l):
    """
    >>> list( cyc_neighbors(list(range(10))) )
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
        return np.array(list(zip(l, np.roll(l, -1, axis = 0))))

class CV:
    """
    Yield cross-validation ranges.

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

