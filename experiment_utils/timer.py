#!/usr/bin/env python

import time
from collections import namedtuple, defaultdict
from typing import List, Tuple
from functools import wraps

from .text import format_table

class Timer:
    """
    TODO: get_stats() probably doesn't handle recursive functions correctly currently

    Context manager for timing execution of code blocks
    >>> with Timer("frobnizing"):
    ...     a = 1 + 1

    >>> r = Timer.events
    >>> len(r)
    2
    >>> r[0].event, r[0].name, r[0].level, r[0].t # doctest: +ELLIPSIS
    ('START', 'frobnizing', 0, ...)
    >>> r[1].event, r[1].name, r[1].level, r[1].t # doctest: +ELLIPSIS
    ('STOP', 'frobnizing', 0, ...)
    >>> list(Timer.get_log()) # doctest: +ELLIPSIS
    ['frobnizing...', 'frobnizing done (...s)']
    >>> list(Timer.get_stats()) # doctest: +ELLIPSIS
    [('frobnizing', ..., 1)]
    """
    Event = namedtuple('Event', ('event', 'name', 't', 'level'))

    START = 'START'
    STOP = 'STOP'
    level = 0
    events: List[Event] = []

    def __init__(self, name):
        self.name = name

    def __call__(self, f):
        # Be a decorator
        @wraps(f)
        def new_f(*args, **kws):
            with self:
                return f(*args, **kws)
        return new_f

    @classmethod
    def get_log(cls):
        import sys
        stack = []
        for event in cls.events:
            if event.event == cls.START:
                stack.append(event)
                yield cls.format_message(event.level, '{}...'.format(event.name))
            elif event.event == cls.STOP:
                assert stack[-1].name == event.name
                yield cls.format_message(event.level, '{} done ({:.3f}s)'.format(event.name, event.t - stack[-1].t))
                stack.pop()
        assert not len(stack)

    @classmethod
    def format_stats(cls, stats = None):
        if stats is None:
            stats = tuple(cls.get_stats())
        return format_table(stats, ('name', 'tot', 'calls'))

    @classmethod
    def get_stats(cls):
        stack = []
        tot_time = defaultdict(float)
        calls = defaultdict(int)

        for event in cls.events:
            if event.event == cls.START:
                stack.append(event)
            elif event.event == cls.STOP:
                assert stack[-1].name == event.name
                tot_time[event.name] += event.t - stack[-1].t
                calls[event.name] += 1
                stack.pop()
        # assert not len(stack)

        l = list(tot_time.items())
        l.sort(key = lambda kv: -kv[1])
        for name, tot in l:
            yield (name, tot, calls[name])

    @staticmethod
    def format_message(level, msg):
        return '  ' * level + msg

    #def log_message(self, msg):
        #s = self.format_message(msg)
        #if callable(self.log_callback):
            #self.log_callback(s)
        #Timer.log.append(s)

    def __enter__(self):
        Timer.events.append(
            Timer.Event(event = Timer.START, name = self.name,
                t = time.time(), level = Timer.level)
        )
        #self.log_message('{}...'.format(self.name))
        #self.t = time.time()
        Timer.level += 1

    def __exit__(self, *args):
        Timer.level -= 1
        Timer.events.append(
            Timer.Event(event = Timer.STOP, name = self.name,
                t = time.time(), level = Timer.level)
        )
        #self.log_message('{} done ({:.3f})'.format(self.name, time.time() - self.t))

