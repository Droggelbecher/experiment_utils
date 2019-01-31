
from typing import Sequence, Iterable, Union, Sequence, Callable, Any
from collections import namedtuple
import numpy as np

import sys
# Hack to fix "typing_extensions not found" on OSX when running this with anything other than mypy
sys.path.extend([
    "/usr/local/Cellar/mypy/0.560_1/libexec/lib/python3.6/site-packages",
    "/usr/local/Cellar/mypy/0.560_1/libexec/vendor/lib/python3.6/site-packages"
    ])
from typing_extensions import Protocol

from .conversions import make_like


class Point(Protocol):
    x : float
    y : float
Path = Sequence[Point]

class TimedPoint(Point):
    t : float
TimedPath = Sequence[TimedPoint]


def bernstein(n: int, k: int):
    """
    Return a function bpoly(x) that is the bernstein polynomial for (n, k)
    """
    from scipy.special import binom
    coeff = binom(n, k)
    def bpoly(x):
        r = coeff * x**k * (1 - x)**(n - k)
        return r
    return bpoly

def bezier(path: Sequence[Any]):
    """
    path: Path to interpolate

    >>> path = [ (0, 0), (1, 1), (5, 1), (6, 2) ]
    >>> f = bezier(path)
    >>> f(0)
    array([[0., 0.]])
    >>> f([0, 1])
    array([[0., 0.],
           [6., 2.]])
    """
    n = len(path)
    path1 = np.array(path)

    def f(t):
        t = np.array(t)
        a = sum(
            np.outer(bernstein(n - 1, i)(t), p)
            for i, p in enumerate(path1)
        )
        return make_like(path, a)
    return f

