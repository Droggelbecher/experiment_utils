#!/usr/bin/env python

import sys
from tracking_pb2 import Track, TrackEntry


def read_data(filename):
    """
    filename: filename of a curfer -data file.
    Yields TrackEntry objects
    """
    file_ = open(filename, 'rb')
    while True:
        block_length_s = file_.read(2)

        if len(block_length_s) < 2:
            break

        block_length = (ord(block_length_s[1]) << 8) | ord(block_length_s[0])
        block = file_.read(block_length)

        entry = TrackEntry()
        entry.ParseFromString(block)

        yield entry


def generate_ttp(trace):
    """
    trace: iterable over curfer TrackEntry objects.
    returns: TTP suitable for positioning & mapmatching as string.
    """

    # For TTP spec see
    # https://confluence.tomtomgroup.com/display/POS/How+to+read+ttp+log+files

    r = 'BEGIN:ApplicationVersion=TomTom Positioning 0.2\n'
    r += '0.000,245,0,SENSOR=Location,periodMS=1000,X=0,Y=0,Z=0,oriX=0,oriY=0,oriZ=0,accuracy=1000000.000000,offset=0.000000,sensitivity=1.000000,min=-1000.000000,max=1000.000000\n'

    startdate = None
    for e in trace:
        if startdate is None:
            startdate = float(e.dateInMillis)

        l = '{},245,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            (float(e.dateInMillis) - startdate) / 1000.0,
            0, # channel
            e.longitude,
            float(e.gpsVerticalAccuracy),
            e.latitude,
            float(e.gpsHorizontalAccuracy),
            e.gpsAltitude,
            1.0, # altitude accuracy
            e.gpsCourse, # heading
            1.0, # heading accuracy
            e.gpsSpeed,
            1.0, # speed accuracy
            1.0, # slope
            1.0, # slope accuracy
            e.distance,
            1.0, # distance accuracy
            e.dateInMillis / 1000.0,
            3, # source of data (Unknown=0, Network=1, GNSS=3, GNSS_LSQ=5)
            1 # GNSS used?
            )


        r += l
    r += 'END\n'
    return r


