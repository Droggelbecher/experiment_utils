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

@cached(
        filename_kws = ['curfer_filename'],
        ignore_kws = ['ttp_filename']
        )
def run_positioning(ttp_filename, curfer_filename):
    # clear log dir
    osutil.rmtree(POSITIONED_TTP_DIR)

    os.chdir(NAVKIT_RUNDIR)
    proc = subprocess.Popen(['./Player', '-c', POSITIONING_XML_FILENAME, '-l', ttp_filename, '-s', '0'])

    ret = proc.wait()

    # If nobody intervened, there *should* be only exactly 1 file now in the log dir
    return os.path.join(POSITIONED_TTP_DIR, os.listdir(POSITIONED_TTP_DIR)[0])

def prepare_mapmatching():
    os.chdir(NAVKIT_RUNDIR)
    osutil.mkdir(NAVKIT_RUNDIR, '/home')
    shutil.copy(PATHMATCHERSETTINGS, 'home/')

@cached(
        filename_kws = ['curfer_filename'],
        ignore_kws = ['ttp_filename']
        )
def run_mapmatching(ttp_filename, curfer_filename):
    os.chdir(NAVKIT_RUNDIR)
    proc = subprocess.Popen(['./OfflineMapMatcherApp', '--input', ttp_filename,
        '--output', MAPMATCHED_TTP_FILENAME])
    ret = proc.wait()
    return MAPMATCHED_TTP_FILENAME
