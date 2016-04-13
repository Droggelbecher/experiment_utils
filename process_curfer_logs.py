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

np.set_printoptions(threshold=99999,linewidth=99999,precision=3)

NAVKIT_DIR = '/home/henning/repos/gitp4/navkit/dev-guidance'
NAVKIT_RUNDIR = NAVKIT_DIR + '/Build/Output/Binary/x86-Linux/Debug/bin'

PATHMATCHERSETTINGS = NAVKIT_DIR + '/Engines/MapMatching/PathMatcher/Matlab/MapMatching/pathmatchersettings.ini'

GPS_TTP_FILENAME = '/tmp/gps.ttp'
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

def run_positioning(ttp_filename):
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

def run_mapmatching(ttp_filename):
    os.chdir(NAVKIT_RUNDIR)
    proc = subprocess.Popen(['./OfflineMapMatcherApp', '--input', ttp_filename,
        '--output', MAPMATCHED_TTP_FILENAME])
    ret = proc.wait()
    return MAPMATCHED_TTP_FILENAME

@cached(filename_kws = ('curfer_filename',))
def curfer_to_road_ids(curfer_filename):
    trace = curfer.read_data(curfer_filename)

    gps_ttp = curfer.generate_ttp(trace)
    with open(GPS_TTP_FILENAME, 'w') as f:
        f.write(gps_ttp)

    positioned_ttp_filename = run_positioning(GPS_TTP_FILENAME)
    mapmatched_ttp_filename = run_mapmatching(positioned_ttp_filename)
    path, road_id_to_endpoints = ttp.extract_roadids(mapmatched_ttp_filename)
    return path, road_id_to_endpoints

if __name__ == '__main__':
    # Precondition: make sure map service runs
    # and needed navkit components are compiled

    prepare_positioning()
    prepare_mapmatching()

    filenames = osutil.find_recursive(sys.argv[1], 'data')

    routes = []
    gps_routes = []

    road_ids_to_endpoints = {}

    for curfer_filename in filenames:
        gps_route, r = curfer_to_road_ids(curfer_filename = curfer_filename)
        # r: road_id => ( (enter_lat, enter_lon), (leave_lat, leave_lon) )
        road_ids_to_endpoints.update(r)
        routes.append(r.keys())
        gps_routes.append(gps_route)

    ids, means, cov = route_analysis.roadid_covariance_matrix(routes)

    w, v_ = LA.eig(cov)

    order = np.argsort(-w)

    #v = v[order]
    v = v_[:,order]

    print("routes")
    print(routes)

    print("road ids sorted")
    print(ids)

    print("means")
    print(means)

    print("cov")
    print(cov)

    print("w")
    print(w)

    print("v")
    print(v)

    print("road_ids_to_endpoints")
    print(road_ids_to_endpoints)

    print("v.shape=", v.shape)
    print("w.shape=", w.shape)


    heatmap = [
            (road_ids_to_endpoints[ids[x]][0][0], road_ids_to_endpoints[ids[x]][0][1], val)
            for x, val in enumerate(means) if val > 0
            ]
    print("heatmap=", heatmap)
    g = gmaps.generate_gmaps(heatmap = heatmap, center = road_ids_to_endpoints[ids[0]][0]) # trips = gps_routes)
    f = open('/tmp/gmaps_mean.html', 'w')
    f.write(g)
    f.close()


    for i_e in range(0, v.shape[1]):
        # one of the eigenvectors (they are not sorted!)
        e = v[:,i_e] # vector of road-id indices

        heatmap = [
                (road_ids_to_endpoints[ids[x]][0][0], road_ids_to_endpoints[ids[x]][0][1], val)
                for x, val in enumerate(e) if val > 0
                ]
        g = gmaps.generate_gmaps(heatmap = heatmap, center = road_ids_to_endpoints[ids[0]][0]) # trips = gps_routes)
        f = open('/tmp/gmaps_{}.html'.format(i_e), 'w')
        f.write(g)
        f.close()



    #for i in range(len(ids)):
        #print(i, cov[i, i])


