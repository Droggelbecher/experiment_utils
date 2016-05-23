#!/usr/bin/env python

import sys
import os
import os.path
import shutil
import subprocess

import curfer
import ttp
import route_analysis
from cache import cached
import osutil
import gmaps

import numpy as np
import numpy.linalg as LA


NAVKIT_DIR = '/home/henning/repos/gitp4/navkit/dev-guidance'
NAVKIT_RUNDIR = NAVKIT_DIR + '/Build/Output/Binary/x86-Linux/Debug/bin'

PATHMATCHERSETTINGS = NAVKIT_DIR + '/Engines/MapMatching/PathMatcher/Matlab/MapMatching/pathmatchersettings.ini'
POSITIONED_TTP_DIR = NAVKIT_RUNDIR + '/positioning_logs'
MAPMATCHED_TTP_FILENAME = '/tmp/mapmatched.ttp'

POSITIONING_XML_FILENAME = '/tmp/positioning.xml'
POSITIONING_XML = '''
<?xml version="1.0" ?>
<Positioning>
    <Logging enabled="true" manual="false" path="{}"/>
</Positioning>
'''.format(POSITIONED_TTP_DIR)

def prepare_positioning():
    """
    Has to be run only once before potentially many calls to run_positioning
    """
    f = open(POSITIONING_XML_FILENAME, 'w')
    f.write(POSITIONING_XML)
    f.close()

@cached()
def run_positioning(ttp):
    """
    TODO TODO TODO

    curfer_rel_filename: filename of data source relative to DATA_SOURCE_DIR
    ttp: TTP content that is to be positioned

    return: positioned TTP as string

    curfer_rel_filename is only used for cache identification!
    """
    TTP_FILENAME = '/tmp/run_positioning.ttp'

    with open(TTP_FILENAME, 'w') as f:
        f.write(ttp)

    # clear log dir
    osutil.rmtree(POSITIONED_TTP_DIR)

    os.chdir(NAVKIT_RUNDIR)
    proc = subprocess.Popen(['./Player', '-c', POSITIONING_XML_FILENAME, '-l', TTP_FILENAME, '-s', '0'])

    ret = proc.wait()

    # If nobody intervened, there *should* be only exactly 1 file now in the log dir
    with open( os.path.join(POSITIONED_TTP_DIR, os.listdir(POSITIONED_TTP_DIR)[0]), 'r' ) as f:
        r = f.read()
    return r

def prepare_mapmatching():
    os.chdir(NAVKIT_RUNDIR)
    osutil.mkdir(NAVKIT_RUNDIR, '/home')
    shutil.copy(PATHMATCHERSETTINGS, 'home/')

@cached()
def run_mapmatching(ttp):
    """
    curfer_rel_filename: filename of data source relative to DATA_SOURCE_DIR
    ttp: positianed TTP content that is to be mapmatched

    return: mapmatched TTP as string
    """
    TTP_FILENAME = '/tmp/run_mapmatching.ttp'

    with open(TTP_FILENAME, 'w') as f:
        f.write(ttp)

    os.chdir(NAVKIT_RUNDIR)
    proc = subprocess.Popen(['./OfflineMapMatcherApp', '--input', TTP_FILENAME,
        '--output', MAPMATCHED_TTP_FILENAME])
    ret = proc.wait()
    with open(MAPMATCHED_TTP_FILENAME, 'r') as f:
        r = f.read()
    return r

