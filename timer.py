#!/usr/bin/env python

import logging
import time


class Timer:
    """
    Context manager for timing execution of code blocks
    >>> with Timer("frobnizing"):
    ...     a = 1 + 1 
    >>> Timer.log # doctest: +ELLIPSIS
    ['frobnizing...', 'frobnizing done (...)']
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

