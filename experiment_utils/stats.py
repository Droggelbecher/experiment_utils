
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

def outliers(a, f=1.5):
    import numpy as np
    Q1, Q3 = np.percentile(a, q=[25, 75])
    iqr = Q3 - Q1
    min_, max_ = Q1 - f * iqr, Q3 + f * iqr
    return ((a < min_) | (a > max_))


def print_stats(a, w=80, h=10):
    import numpy as np
    import sys

    min_ = np.min(a)
    max_ = np.max(a)
    avg = np.average(a)
    avg_idx = int(w * (avg - min_) / (max_ - min_)) - 1

    bins = np.linspace(min_, max_, w)
    hist, _ = np.histogram(a, bins=bins)
    hist = hist.astype('f8')
    hist *= float(h) / np.max(hist)

    print()
    for hlim in range(h, 0, -1):
        for i, hhh in enumerate(hist):
            sys.stdout.write('#' if hhh >= hlim else ' ')
        sys.stdout.write('\n')

    sys.stdout.write(' ' * avg_idx)
    sys.stdout.write('^\n')
    sys.stdout.write('<' + ' ' * (w - 2) + '>\n')


    print(f'shape={a.shape}')
    print(f'min={min_} avg={avg} max={max_}')
    print(f'zeros={np.sum(a == 0)} nans={np.sum(np.isnan(a))}')

    qs = [1, 10, 25, 50, 75, 90, 99]
    perc = np.percentile(a, q=qs)
    print()
    print('Percentiles:')
    for q, p in zip(qs, perc):
        print(f' {q:02d}%: {p}')



