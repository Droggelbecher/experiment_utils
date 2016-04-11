#!/usr/bin/env python

import os
import os.path
import shutil


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

