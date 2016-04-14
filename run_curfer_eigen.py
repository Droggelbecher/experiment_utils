#!/usr/bin/env python

import sys
import os
import os.path
import shutil
import subprocess
import numpy as np
import numpy.linalg as LA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import math

import curfer
import ttp
import route_analysis
from cache import cached
import osutil
import gmaps
from timer import Timer
import plots
from navkit import prepare_positioning, prepare_mapmatching, run_positioning, run_mapmatching

np.set_printoptions(threshold=99999,linewidth=99999,precision=3)


GPS_TTP_FILENAME = '/tmp/gps.ttp'
MAX_EIGENVECTORS = 50

@cached(filename_kws = ['curfer_filename'])
def curfer_to_road_ids(curfer_filename):
    """
    Convert the given curfer trace to TTP,
    then use NavKit to position & mapmatch it and extract road ids from the result.
    Makes heavy use of caching to minimize navkit executions.
    Intermediate files will only exist temporarily (filenames are being reused!)
    """

    trace = curfer.read_data(curfer_filename)

    gps_ttp = curfer.generate_ttp(trace)
    with open(GPS_TTP_FILENAME, 'w') as f:
        f.write(gps_ttp)

    @cached(
            filename_kws = ['curfer_filename'],
            ignore_kws = ['mapmatched_ttp_filename']
            )
    def get_road_ids(mapmatched_ttp_filename, curfer_filename):
        return ttp.extract_roadids(mapmatched_ttp_filename)

    positioned_ttp_filename = run_positioning(
            ttp_filename = GPS_TTP_FILENAME,
            curfer_filename = curfer_filename
            )
    mapmatched_ttp_filename = run_mapmatching(
            ttp_filename = positioned_ttp_filename,
            curfer_filename = curfer_filename
            )
    path, road_id_to_endpoints = get_road_ids(
            mapmatched_ttp_filename = mapmatched_ttp_filename,
            curfer_filename = curfer_filename
            )
    return path, road_id_to_endpoints


def render_road_ids(endpoints, filename, info = {}):
    """
    endpoints = [
        ( (lat, lon), (lat, lon), weight )
    ]
    """
    trips = [ [from_, to] for from_, to, w in endpoints ]
    #trip_weights = [ math.tanh(w * 10) for from_, to, w in endpoints ]
    trip_weights = [ w for from_, to, w in endpoints ]
    g = gmaps.generate_gmaps(center = endpoints[0][0], trips = trips, trip_weights = trip_weights, info = info)

    f = open(filename, 'w')
    f.write(g)
    f.close()

def preprocess_data(curfer_directory):
    # Precondition: make sure map service runs
    # and needed navkit components are compiled

    prepare_positioning()
    prepare_mapmatching()

    filenames = osutil.find_recursive(sys.argv[1], 'data')

    routes = []
    gps_routes = []

    road_ids_to_endpoints = {}

    # Read in / convert all the curfer files
    # ---> routes and road_id => endpoint conversion

    for curfer_filename in filenames:
        gps_route, r = curfer_to_road_ids(curfer_filename = curfer_filename)
        # r: road_id => ( (enter_lat, enter_lon), (leave_lat, leave_lon) )
        road_ids_to_endpoints.update(r)
        routes.append(r.keys())
        gps_routes.append(gps_route)

    return routes, gps_routes, road_ids_to_endpoints


def plot_pc2(a, filename):
    """
    routes: np array, rows: routes, columns: road_ids, elems in {0, 1}
    """

    # 2 2d suplots

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(a[:,0], a[:,1])

    ax = fig.add_subplot(212)
    ax.scatter(a[:,0], a[:,2])
    plt.savefig(filename)


if __name__ == '__main__':
    routes, gps_routes, road_ids_to_endpoints = preprocess_data(sys.argv[1])

    # Generate road id covariance matrix and calculate its eigenvectors

    ids, means, cov = route_analysis.roadid_covariance_matrix(routes)
    w, v = LA.eig(cov)

    # w: eigenvalues, v: eigenvectors
    order = np.argsort(-w)
    v = v[:,order]
    w = w[order]

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

    with Timer('routes to matrix'):
        a = route_analysis.routes_to_array(routes, ids)


    # Plot routes on PCs
    #
    plot_pc2(np.dot(a, v[:,:3]), '/tmp/pca.pdf')

    pca = PCA(n_components=10)
    S_pca_ = pca.fit(a).transform(a)

    plot_pc2(S_pca_, '/tmp/pca2.pdf')



    print("---a=")
    print(a.shape)
    print(a)


    with Timer('ICA'):
        ica = FastICA(
                max_iter=2000,
                n_components=10,
                fun='exp',
                ) #max_iter=20000,
                #tol = 0.000001,
                #fun='cube',
                #n_components=3)
        #print(a)
        S_ica_ = ica.fit(a).transform(a)  # Estimate the sources
        #print(ica.get_params())
        #S_ica_ /= S_ica_.std(axis=0)

    plot_pc2(S_ica_, '/tmp/ica.pdf')
    plots.all_relations(S_ica_)

    print("ica_=", ica.components_.shape)
    print("ica_ components", ica.components_.T)
    print("ica_ mixing", ica.mixing_)
    print("ica_ mean", ica.mean_)
    print("ica_ whitening", ica.whitening_)
    del v
    v = ica.components_.T
    #v = ica.mixing_

    #v = S_ica_


    # Render mean
    #

    endpoints = [
            ((road_ids_to_endpoints[ids[x]][0][0], road_ids_to_endpoints[ids[x]][0][1]),
             (road_ids_to_endpoints[ids[x]][1][0], road_ids_to_endpoints[ids[x]][1][1]),
             val)
            for x, val in enumerate(means)
            ]

    render_road_ids(endpoints, '/tmp/gmaps_mean.html')

    # Render Eigenvectors

    for i_e in range(0, min(MAX_EIGENVECTORS, v.shape[1])):
        # one of the eigenvectors (they are not sorted!)
        e = v[:,i_e] # vector of road-id indices
        print("e.shape=", e.shape)

        endpoints = [
                ((road_ids_to_endpoints[ids[x]][0][0], road_ids_to_endpoints[ids[x]][0][1]),
                 (road_ids_to_endpoints[ids[x]][1][0], road_ids_to_endpoints[ids[x]][1][1]),
                 float(val))
                for x, val in enumerate(e)
                ]

        render_road_ids(endpoints, '/tmp/gmaps_{}.html'.format(i_e), info = { 'Eigenvalue': w[i_e] })




