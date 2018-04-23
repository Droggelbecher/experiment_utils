
from collections import Counter

from . import text

class Stats:

    """
    >>> s = Stats(binwidth = 10)
    >>> s.push(3)
    >>> s.push(67)
    >>> s.push(5)
    >>> s.push(8)
    >>> print(s)
    min=3 avg=20.75 max=67
    0 60
    3 1
    <BLANKLINE>
    """

    def __init__(self, binwidth):
        self.n = 0
        self.sum = 0
        self.min = None
        self.max = None
        self.binwidth = binwidth
        self.histogram = Counter()

    def push(self, value):
        bin = int(value // self.binwidth)
        self.histogram[bin] += 1
        self.n += 1
        self.sum += value
        self.min = value if self.min is None else min(value, self.min) 
        self.max = value if self.max is None else max(value, self.max)

    @property
    def average(self):
        return self.sum / self.n

    def __str__(self):
        r = f'min={self.min} avg={self.average} max={self.max}\n'

        items = sorted(self.histogram.items())
        r += text.format_table(
            zip(*[(k * self.binwidth, v) for k, v in items])
            )
        return r

