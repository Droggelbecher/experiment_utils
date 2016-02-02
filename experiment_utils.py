#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set fdm=indent foldenable:

from itertools import islice, cycle
import time
import math
import numpy as np

import logging

def free_ram():
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024


class Timer:
    """
    Context manager for timing execution of code blocks
    >>> with Timer("frobnizing"):
    ...     a = 1 + 1 # doctest: +ELLIPSIS
    frobnizing...
     frobnizing done (...s)
    """
    log = []
    level = 0
    def __init__(self, name):
        self.name = name

    @classmethod
    def pop_log(self):
        r = self.log
        self.log = []
        return r

    def log_message(self, msg):
        s = '  ' * Timer.level + msg
        logging.debug(s)
        Timer.log.append(s)

    def __enter__(self):
        self.log_message('{}...'.format(self.name))
        self.t = time.time()
        Timer.level += 1

    def __exit__(self, *args):
        Timer.level -= 1
        self.log_message('{} done ({:.3f})'.format(self.name, time.time() - self.t))

class Summary:
    def __init__(self, **kws):
        self.__dict__.update(kws)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

class Situation:
    def __init__(self, name, **kws):
        self.name = name
        self.d_min = 0
        self.optional = False
        self.__dict__.update(kws)

    def match(self, ts, vs, i):
        history = SignalHistory(100)
        t0 = ts[i]
        j = 0

        t_report = t0 + 100.0

        print("{} match t0={}".format(self.name, t0))

        for t, v in zip(ROSlice(ts, i), ROSlice(vs, i)):
            if t >= t_report:
                print("match {} t={}".format(self.name, t))
                t_report = t + 100.0

            if hasattr(self, 'd_max') and t - t0 > self.d_max:
                print("dmax")
                break

            # Value too high
            if hasattr(self, 'max') and v > self.max:
                print("max", v, "sp", history.span(), "t", t, "t0", t0)
                break

            history.push(t, v)
            j += 1

            if hasattr(self, 'min') and v < self.min:
                print("min", v)
                break

        if j == 0 or (hasattr(self, 'd_min') and history.span() < self.d_min) \
                and not self.optional:
            print("!match {} min={} max={} j={} dmin={} span={}".format(self.name, getattr(self,
                'min', None), getattr(self, 'max', None), j, getattr(self,
            'd_min', None), history.span()))
            return None

        return Summary(
                situation = self,
                n = j,
                i = i + j,
                t0 = t0,
                d = history.span(),
                v_sum = history.sum(),
                v_average = history.average()
        )

    def find(self, ts, vs, i):
        """
        Like match() but allow (and ignore) non-matching data before the match.
        If you specify d_min it will match non-greedy, that is a span of just
        d_min.
        """
        history = SignalHistory(100)
        t0 = ts[i]
        j = 0
        match_start = None
        print("      find t0 {}".format(t0))

        assert not hasattr(self, 'd_max'), 'd_max not supported in find()'

        for t, v in zip(ROSlice(ts, i), ROSlice(vs, i)):
            matching = (
                    (not hasattr(self, 'max') or v <= self.max) and
                    (not hasattr(self, 'min') or v >= self.min)
            )
            j += 1

            if match_start is None:
                if matching:
                    print("      find match start {}".format(t))
                    match_start = t
                    history.push(t, v)
            else:
                history.push(t, v)

                if not hasattr(self, 'd_min') or (t - match_start) >= self.d_min:
                    print("      find success ({} {})".format(match_start, t))
                    return Summary(
                            situation = self, n = j, i = i + j,
                            t0 = match_start,
                            d = history.span(),
                            v_sum = history.sum(),
                            v_average = history.average()
                    )

                if not matching:
                    print("      find match end {}".format(t))
                    match_start = None
                    history = SignalHistory(100)

        print("    no find()")
        return None


    def match__(self, ts, vs, i):
        t0 = ts[i]
        j = 0
        t_ = t0
        vsum = 0.0
        print("match " + self.name + " t0=" + str(t0))
        for t, v in zip(ROSlice(ts,i), ROSlice(vs,i)):
            if hasattr(self, 'd_max') and t - t0 > self.d_max:
                print("dmax")
                break

            # Value too high

            if hasattr(self, 'max') and v > self.max:
                if self.outliers_above < self.max_outliers_above:
                    self.outliers_above += 1
                else:
                    print("max", v)
                    t_ = t
                    break
            else:
                #if hasattr(self, 'max'):
                    #print(v, "<=", self.max)
                self.outliers_above = 0

            j += 1
            vsum += v * (t - t_)
            t_ = t

            # Value too low

            if hasattr(self, 'min') and v < self.min:
                if self.outliers_below < self.max_outliers_below:
                    self.outliers_below += 1
                else:
                    print("min")
                    t_ = t
                    break
            else:
                self.outliers_below = 0

        if j == 0 or (hasattr(self, 'd_min') and t_ - t0 < self.d_min):
            print("j=",j,"t_=", t_, "t0=",t0, "dmin=", self.d_min)
            return None

        return Summary(
                situation = self,
                n = j,
                i = i + j,
                t0 = t0,
                d = t_ - t0,
                v_mean =  vsum / (t_ - t0)
        )

class Repeat:
    def __init__(self, *args):
        self.situations = args

    def __iter__(self):
        print("EM: repeat iter")
        return cycle(self.situations)


"""
    >>> em = ExperimentModel(
    ...     Situation('before', max=1.1),
    ...     Repeat(
    ...         Situation('prep', min=1.2, d_min=0.01, d_max=0.1),
    ...         Situation('between', max=1.1, d_min=0.05, d_max=0.08),
    ...         Situation('exp', min=1.2, d_min=0.01, d_max=0.1),
    ...         Situation('after', max=1.1, d_min=0.4, d_max=0.6),
    ...     )
    ... )
    >>> ts = [0.001 * x for x in range(10000)]
    >>> vs = [0.8] * 99 + [1.3] * 20 + [1.0] * 60 + [1.4] * 100 + [1.0] * 505
    >>> ts, vs = align(ts, vs)
    >>> l = list(em.match(ts, vs))
    >>> [x.situation.name for x in l]
    ['before', 'prep', 'between', 'exp', 'after']
"""

class ExperimentModel:
    """
    Models the process of an experiment
    
    >>> em = ExperimentModel(
    ...     Situation('before', max=1.1),
    ...     Repeat(
    ...         Situation('prep', min=1.2, d_min=0.01, d_max=0.1),
    ...         Situation('between', max=1.1, d_min=0.05, d_max=0.08),
    ...         Situation('exp', min=1.2, d_min=0.01, d_max=0.1),
    ...         Situation('after', max=1.1, d_min=0.4, d_max=0.6),
    ...     )
    ... )
    """
    def __init__(self, *args):
        self.situations = args

    def __iter__(self):
        #yield from ExperimentModel.iterate_situations(self.situations)
        for x in ExperimentModel.iterate_situations(self.situations):
            yield x

    @staticmethod
    def iterate_situations(situations):
        for s in situations:
            if hasattr(s, '__iter__'):
                for x in iter(s):
                    yield x
            else:
                yield s

    def match(self, ts, vs, situations = None, i = 0):
        if situations is None:
            situations = self.situations

        #for situation in situations:
        for situation in ExperimentModel.iterate_situations(situations):
            print("EM: matching {}".format(situation.name))
            r = situation.match(ts, vs, i)
            if r is None:
                print("no match")
                #raise StopIteration
                break
            #else:
            yield r
            i = r.i
            if i >= len(ts):
                print("all ts eaten")
                #raise StopIteration
                break

    def find(self, ts, vs):
        i = 0

        s = self.situations[0]
        r = s.find(ts, vs, i)
        if r is None:
            print("no find")
            return
        yield r
        i = r.i
        if i >= len(ts):
            print("all ts eaten in find")
            return
        #yield from self.match(ts, vs, situations=self.situations[1:], i=i)
        for x in self.match(ts, vs, situations=self.situations[1:], i=i):
            yield x


def align(l1, l2):
    """
    Return slices of both sequences, shortened to the length
    of the shorter one.
    >>> align(range(5), range(10))
    (range(0, 5), range(0, 5))
    >>> align("foobar", [1, 2, 3, 4])
    ('foob', [1, 2, 3, 4])
    """
    return l1[:len(l2)], l2[:len(l1)]


class ROSlice:
    def __init__(self,list_,i,j=None):
        self.list_ = list_
        self.current = i
        self.size = j or len(list_)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.size:
            raise StopIteration
        val = self.list_[self.current]
        self.current += 1
        return val

def shift(l, n):
    """
    >>> shift( (0, 1, 2, 3), 1 )
    (1, 2, 3, 0)
    >>> shift( (0, 1, 2, 3), -1 )
    (3, 0, 1, 2)
    """
    return l[n:] + l[:n]


def quantize(it, k):
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
    rv = []
    rt = []
    i = 0
    s = 0
    t_ = 0
    for t, v in it:
        if i == k:
            yield (t_, s / i)
            #rv.append(s / i)
            #rt.append(t_)
            t_ = t
            s = 0
            i = 0
        s += v
        i += 1

    if i > 0:
        yield (t_, s / i)
        #rv.append(s / i)
        #rt.append(t_)

    #return zip(rt, rv)

def t_quantize(it, delta_t, op=sum, align=0, align_phase=0):
    """
    # ts can be floats as well but its horrible to doctest
    >>> ts = [100, 199, 290, 300, 550, 701, 702, 750, 751, 752, 800, 900, 1000]
    >>> vs = [  1,   2,   3,   4,   5,    6,    7,   8,    9,   10, 1, 1, 1]
    >>> ts2, vs2 = zip(*t_quantize(zip(ts, vs), 100.0))
    >>> ts2
    (100, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0)
    >>> vs2
    (3, 3, 4, 0, 5, 0, 40, 1, 1, 1)

    >>> ts = [190, 210, 310]
    >>> vs = [1, 1, 1]
    >>> ts2, vs2 = zip(*t_quantize(zip(ts, vs), 100.0))
    >>> ts2
    (190, 290.0)
    >>> vs2
    (2, 1)

    >>> ts2, vs2 = zip(*t_quantize(zip(ts, vs), 100.0, align=100.0))
    >>> ts2
    (100.0, 200.0, 300.0)
    >>> vs2
    (1, 1, 1)

    >>> ts2, vs2 = zip(*t_quantize(zip(ts, vs), 100.0, align=100.0, align_phase=-20))
    >>> ts2
    (180.0, 280.0)
    >>> vs2
    (2, 1)

    """
    it = materialize(it)
    t0, v0 = it[0]

    if align != 0:
        t0 = align * math.floor((t0 - align_phase) / align) + align_phase

    l = []
    for t, v in it:
        if t >= t0 + delta_t:
            yield t0, op(l)
            l = []
            t0 += delta_t
            while t >= t0 + delta_t:
                yield t0, op(l)
                t0 += delta_t
        l.append(v)

    while t >= t0 + delta_t:
        yield t0, op(l)
        t0 += delta_t
    yield t0, op(l)


def fold(it, k, skip=0):
    """
    Combine every k'th output slot into a list.
    Useful for creating  box plots of repeated experiments.

    >>> ts = range(100)
    >>> vs = list(range(10)) * 10
    >>> ts2, vs2 = zip(*fold(zip(ts, vs), 5))
    >>> ts2 # doctest: +ELLIPSIS
    (0, 1, 2, 3, 4)
    >>> vs2[0] == [0, 5] * 10
    True
    >>> vs2[1] == [1, 6] * 10
    True
    >>> it = [[0, 1],  [10, 2], [20, 2], [30, 3], [40, 2], [50, 7]]
    >>> list(fold(it, 2))
    [(0, [1, 2, 2]), (10, [2, 3, 7])]

    #>>> list(fold(it, 30))
    #[(0, [1]), (10, [2]), (20, [2]), (30, [3]), (40, [2]), (50, [7])]
    """
    it = materialize(it)
    rv = []
    rt = []
    i = 0
    #it = list(it)
    n = k * (len(it) // k)
    #n = len(it) #k * (len(it) // k)
    for t, v in islice(it, n):
        if skip == 0:
            if len(rt) < k:
                rt.append(t)
                rv.append([])
            rv[i].append(v)
        i += 1
        if i == k:
            i = 0
            if skip > 0:
                skip -= 1
    return zip(rt, rv)

def t_fold(it, delta_t, align=0, align_phase=0):
    """
    # ts can be floats as well but its horrible to doctest
    >>> ts = [100, 200, 290, 300, 550, 701, 702, 750, 751, 752]
    >>> vs = [  1,   2,   3,   4,   5,    6,    7,   8,    9,   10]
    >>> ts2, vs2 = zip(*t_fold(zip(ts, vs), 100.0))
    >>> ts2
    (0.0, 0.0, 0.0, 1.0, 2.0, 50.0, 50.0, 51.0, 52.0, 90.0)
    >>> vs2
    (1, 2, 4, 6, 7, 5, 8, 9, 10, 3)

    >>> ts = [190, 210, 309]
    >>> vs = [1, 1, 1]
    >>> ts2, vs2 = zip(*t_fold(zip(ts, vs), 100.0))
    >>> ts2
    (9.0, 10.0, 90.0)
    >>> ts2, vs2 = zip(*t_fold(zip(ts, vs), 100.0, align_phase=-20))
    >>> ts2
    (70.0, 89.0, 90.0)
    """
    # This can be improved to run in O(n) instead O(n log n)

    #it = list(it)
    #t = it[0][0]
    #print(t, t + align_phase, (t+align_phase) % delta_t)

    l = [((t + align_phase) % delta_t, v) for t, v in it]
    l.sort()
    return l



def make_sequence(s):
    t = type(s)
    if t in (list, tuple): return s
    return list(s)

def materialize(s):
    #if type(s) in (iter, list, tuple, list_iterator):
    if hasattr(s, '__iter__'):
        return [
            materialize(x) for x in s
        ]
    else:
        return s


def join_boxes_values(*args):
    """
    >>> join_boxes_values( [(0,1), [7,9], range(3)], [(1,7,9), range(2), range(2)] )
    [[0, 1, 1, 7, 9], [7, 9, 0, 1], [0, 1, 2, 0, 1]]
    >>> join_boxes_values( [[1, 2], [10, 0], [1, 2, 3, 4]], [[3], [], [6, 7]] )
    [[1, 2, 3], [10, 0], [1, 2, 3, 4, 6, 7]]
    """
    args = make_sequence(args)
    r = [list(x) for x in args[0][:]]
    #assert type(r[0]) is list
    for a in args[1:]:
        for x, y in zip(r, a):
            x.extend(make_sequence(y))
    return r

    #return ((tuple(tuple(items[0])[0], sum((list(tuple(x)[1]) for x in items), [])) for items in zip(*args))

def join_boxes(*args):
    """
    >>> a = [(0, [1,2]), (1, [10,0]), (2, [1,2,3,4])]
    >>> b = [(0, [3]), (1, []), (2, [6, 7])]
    >>> list(join_boxes(a, b))
    [(0, [1, 2, 3]), (1, [10, 0]), (2, [1, 2, 3, 4, 6, 7])]
    >>> a = zip(range(3), (iter([x]) for x in range(10, 13)))
    >>> b = zip(range(3), ([x] for x in range(20, 23)))
    >>> list(join_boxes(a, b))
    [(0, [10, 20]), (1, [11, 21]), (2, [12, 22])]
    """
    assert len(args) > 1
    args = materialize(args)
    #return args[0]
    ts = (list(x)[0] for x in args[0])
    vs = tuple( [list(list(x)[1]) for x in a] for a in args )
    j = join_boxes_values(*vs)
    #ts, vs = zip(*args)
    return zip(ts, j)

def median(l):
    """
    A crappy, O(n log n) median implementation.
    """
    if len(l) == 0: return None
    elif len(l) == 1: return l[0]
    #print("{} -> {}".format(l, sorted(l)[int(len(l) / 2)]))
    sl = sorted(l)
    if len(l) % 2:
        return sorted(l)[int(len(l) / 2)]
    return (sl[int(len(l) / 2)] + sl[int(len(l) / 2 - 1)]) / 2.0

def average(l):
    return sum(l)/len(l)


def np_moving_average(a, n=3) :
    """
    Moving average of width n (entries) on a numpy array.

    sauce: http://stackoverflow.com/questions/14313510/moving-average-function-on-numpy-scipy
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    ret[:n] = ret[:n] / (np.arange(n) + 1)
    return ret


def t_average(it, delta_t = .1):
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

    lv = []
    lt = []
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
            yield (t, s / (t - lt[0]))
        else:
            yield (t, float(v))
        t_prev = t

class SignalHistory:
    def __init__(self, duration):
        self.duration = duration
        self.backlog = []
        self.sum_ = 0

    def push(self, t, v):
        if len(self.backlog):
            self.sum_ += v * (t - self.latest()[0])
        self.backlog.append((t, v))
        self.collect_garbage()

    def get(self, t):
        tprev = 0
        for t_, v_ in self.backlog:
            if t >= t_:
                #print(t, ">=", t_)
                return v_
        if len(self.backlog):
            return self.backlog[0][1]
        return 0

    def latest(self):
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


def band_stop(it, T=.1):
    """

    Filters out a frequence band

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


"""
>>> p = PickleCache("/tmp/foo.p", input_files=["bar.txt"])
>>> if p.needs_computation:
...   p.result = 1 + 1
>>> p.result
2
"""
class PickleCache:
    def __init__(self, filename, input_files=()):
        self.filename = filename
        self.input_files = tuple(input_files)
        self._result = None
        self._result_timestamp = None

    def ram_up_to_date(self):
        if self.input_files:
            latest = max(os.getmtime(x) for x in self.input_files)
            return (self._result_timestamp is not None and self._result_timestamp >= latest)
        return self._result_timestamp is not None

    def file_up_to_date(self):
        if not os.path.exists(self.filename):
            return False
        if self.input_files:
            latest = max(os.getmtime(x) for x in self.input_files)
            return os.getmtime(self.filename) >= latest
        return True

    @property
    def needs_computation(self):
        return not self.ram_up_to_date() and not self.file_up_to_date()

    @property
    def result(self):
        if self.ram_up_to_date():
            return self._result
        if self.file_up_to_date():
            self._result = pickle.load(open(self.filename, 'rb'))
            self._result_timestamp = time.time()
            return self._result
        raise Exception('No result available!')

    @result.setter
    def set_result(self, r):
        self._result = r
        self._result_timestamp = time.time()
        pickle.dump(self._result, open(self.filename, 'wb'))






