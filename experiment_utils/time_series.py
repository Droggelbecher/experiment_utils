
from typing import Iterable, Any, Generator, Tuple, List

TimeSeriesElement = Tuple[float, Any]
TimeSeries = Iterable[TimeSeriesElement]

def quantize(it: TimeSeries, k: int) -> TimeSeries:
    """
    Assumes equidistant time slots.
    Combine (average) k consequtive time slots into one, i.e.
    the output sequence will be shorter than the input sequence by factor k.

    >>> ts = range(100)
    >>> vs = list(range(10)) * 10
    >>> ts2, vs2 = zip(*quantize(zip(ts, vs), 10))
    >>> ts2
    (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    >>> len(vs2) == len(ts2)
    True
    >>> all((x == 4.5 for x in vs2))
    True
    """
    i = 0
    s = 0
    t_ = 0.0
    for t, v in it:
        if i == k:
            yield (t_, s / float(i))
            t_ = t
            s = 0
            i = 0
        s += v
        i += 1

    if i > 0:
        yield (t_, s / float(i))

def t_average(it: TimeSeries, delta_t: float = .1) -> TimeSeries:
    """
    Moving average of width delta_t.

    >>> ts = range(40)
    >>> vs = [60] * 10 + [0] * 10 + [20] * 20
    >>> l = list(t_average(zip(ts, vs), 10))
    >>> l
    [(0, 60.0), (1, 60.0), (2, 60.0), (3, 60.0), (4, 60.0), (5, 60.0), (6, 60.0), (7, 60.0), (8, 60.0), (9, 60.0), (10, 54.0), (11, 48.0), (12, 42.0), (13, 36.0), (14, 30.0), (15, 24.0), (16, 18.0), (17, 12.0), (18, 6.0), (19, 0.0), (20, 2.0), (21, 4.0), (22, 6.0), (23, 8.0), (24, 10.0), (25, 12.0), (26, 14.0), (27, 16.0), (28, 18.0), (29, 20.0), (30, 20.0), (31, 20.0), (32, 20.0), (33, 20.0), (34, 20.0), (35, 20.0), (36, 20.0), (37, 20.0), (38, 20.0), (39, 20.0)]
    >>> len(l) == len(ts)
    True

    """

    lv: List[Any] = []
    lt: List[float] = []
    s = 0
    t_prev = None
    for t, v in it:

        # compute number of outdated entries
        outdated = t - delta_t
        j = len(lt)
        ds = 0
        for i in range(len(lt)):
            if lt[i] >= outdated:
                j = i
                break
            else:
                if i < len(lt) - 1:
                    ds += (lt[i + 1] - lt[i]) * lv[i + 1]
        s -= ds
        lt = lt[j:]
        lv = lv[j:]
        lv.append(v)
        lt.append(t)
        if t_prev is not None:
            s += v * (t - t_prev)
        if t > lt[0]:
            yield (t, s / float(t - lt[0]))
        else:
            yield (t, float(v))
        t_prev = t


class SignalHistory:
    """
    A "moving" history of time series signals, keeps the newest signals
    in a time window of specified width.
    """
    def __init__(self, duration):
        self.duration = duration
        self.backlog = []
        self.sum_ = 0

    def push(self, t, v):
        """
        Add a new signal to the history.
        """
        if len(self.backlog):
            self.sum_ += v * (t - self.latest()[0])
        self.backlog.append((t, v))
        self.collect_garbage()

    def get(self, t, default = 0):
        """
        Get oldest available signal from history with a timestamp >= t.
        If no such signal is found, return default.
        """
        for t_, v_ in self.backlog:
            if t >= t_:
                return v_
        return default

    def latest(self):
        """
        Return the newest available signal as (t, v) pair.
        """
        return self.backlog[-1]

    def collect_garbage(self):
        outdated = self.latest()[0] - self.duration
        j = 0 #len(self.backlog)
        ds = 0
        for i in range(len(self.backlog)):
            if self.backlog[i][0] >= outdated:
                j = i
                break
            else:
                if i < len(self.backlog) - 1:
                    ds += (self.backlog[i + 1][0] - self.backlog[i][0]) * self.backlog[i + 1][1]
        self.sum_ -= ds
        self.backlog = self.backlog[j:]

    def span(self):
        if len(self.backlog) == 0:
            return 0
        return self.backlog[-1][0] - self.backlog[0][0]

    def average(self):
        if self.span() == 0:
            return self.sum_
        return self.sum_ / self.span()

    def sum(self):
        return self.sum_


def band_stop(it: TimeSeries, T: float = .1) -> TimeSeries:
    """
    Filters out a frequency band by substraction of past values.
    Assumes timestamps to be sufficiently equidistant.

    it: Iterator over (t, v) pairs (t=timestamp, v=value)
    T: period to filter out.

    This yields (t, v) pairs of the filtered signal.

    >>> import math
    >>>
    >>> ts = range(20)
    >>> vs = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    >>> ts2, vs2 = zip(*band_stop(zip(ts, vs), 4))
    >>> list(vs2)[4:]
    [0, 0, 0, 0, 0, 0, 0, 0, 0]

    This also works for e.g. a sine curve of the right frequency:

    >>> ts = range(200)
    >>> vs = [math.sin(.05 * x * math.pi) for x in range(200)]
    >>> ts2, vs2 = zip(*band_stop(zip(ts, vs), 40))
    >>> list(filter(lambda x: abs(x) > 0.001, list(vs2)[40:]))
    []

    Or even a sine curve of a multiple of the frequency:

    >>> ts = range(200)
    >>> vs = [math.sin(.1 * x * math.pi) for x in range(200)]
    >>> ts2, vs2 = zip(*band_stop(zip(ts, vs), 40))
    >>> list(filter(lambda x: abs(x) > 0.001, list(vs2)[40:]))
    []

    However supposedly not for a fraction of the original frequency
    (which makes sense if you think about it):

    >>> ts = range(200)
    >>> vs = [math.sin(.025 * x * math.pi) for x in range(200)]
    >>> ts2, vs2 = zip(*band_stop(zip(ts, vs), 40))
    >>> len(list(filter(lambda x: abs(x) > 0.001, list(vs2)[40:])))
    156

    """
    h = SignalHistory(duration=T)
    for t, v in it:
        h.push(t, v)
        yield (t, v - h.get(t - T))



