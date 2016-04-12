#!/usr/bin/env python

import os
import os.path
import shutil
import fnmatch

def rmtree(*args):
    p = os.path.join(*args)
    try:
        shutil.rmtree(p)
    except OSError:
        pass

def mkdir(*args):
    p = os.path.join(*args)
    try:
        os.mkdir(p)
    except OSError:
        pass

def find_recursive(directory, pattern):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)

def free_ram():
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024

