
import math
from typing import Sequence, Any, Iterable, List, Generator
import itertools

def align(l1: Sequence[Any], l2: Sequence[Any]) -> Iterable[Any]:
    """
    Return slices of both sequences, shortened to the length
    of the shorter one.
    >>> align(range(5), range(10))
    (range(0, 5), range(0, 5))

    >>> align(list(range(5)), list(range(10)))
    ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

    >>> align("foobar", [1, 2, 3, 4])
    ('foob', [1, 2, 3, 4])
    """
    return l1[:len(l2)], l2[:len(l1)]

def shift(l: Sequence[Any], n: int) -> Iterable[Any]:
    """
    >>> tuple( shift( (0, 1, 2, 3), 1 ) )
    (1, 2, 3, 0)
    >>> tuple( shift( (0, 1, 2, 3), -1 ) )
    (3, 0, 1, 2)
    """
    return itertools.chain(l[n:], l[:n])

def make_sequence(s: Iterable[Any]) -> Sequence[Any]:
    t = type(s)
    if isinstance(t, list) or isinstance(t, tuple):
        return s
    return list(s)

def chunks(list_: Sequence[Any], n: int) -> Generator[Sequence[Any], None, None]:
    """
    Split list-like list_ into n equally sized chunks (the last one may be
    smaller).
    list_ must support len() and sliced access.
    Yields elements.

    >>> l = list(range(10, 75))
    >>> list( chunks(l, 7) )
    [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54, 55, 56, 57, 58, 59], [60, 61, 62, 63, 64, 65, 66, 67, 68, 69], [70, 71, 72, 73, 74]]
    """

    sz = int(math.ceil(len(list_) / float(n)))
    for i in range(0, len(list_), sz):
        yield list_[i:i + sz]


def unique(list_: Iterable[Any]) -> List[Any]:
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(list_)) # type: ignore

