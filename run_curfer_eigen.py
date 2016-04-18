#!/usr/bin/env python

import sys
import os
import os.path
import shutil
import subprocess
import math
from datetime import datetime

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np
import numpy.linalg as LA

from cache import cached
from navkit import prepare_positioning, prepare_mapmatching, run_positioning, run_mapmatching
from timer import Timer
import curfer
import gmaps
import osutil
import plots
import route_analysis
import ttp

np.set_printoptions(threshold=99999,linewidth=99999,precision=3)


GPS_TTP_FILENAME = '/tmp/gps.ttp'
MAX_COMPONENTS = 10

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
    d = get_road_ids(
            mapmatched_ttp_filename = mapmatched_ttp_filename,
            curfer_filename = curfer_filename
            )
    return d


def preprocess_data(curfer_directory):
    # Precondition: make sure map service runs
    # and needed navkit components are compiled

    prepare_positioning()
    prepare_mapmatching()

    filenames = osutil.find_recursive(sys.argv[1], 'data')

    routes = []
    coordinate_routes = []
    departure_times = []
    arrival_times = []
    road_ids_to_endpoints = {}

    # Read in / convert all the curfer files
    # ---> routes and road_id => endpoint conversion

    for curfer_filename in filenames:
        d = curfer_to_road_ids(curfer_filename = curfer_filename)

        road_ids_to_endpoints.update(d['road_ids_to_endpoints'])
        routes.append(d['roadids'])
        coordinate_routes.append(d['coordinates'])
        departure_times.append(d['departure_time'])
        arrival_times.append(d['departure_time'])

        #gps_route, r = curfer_to_road_ids(curfer_filename = curfer_filename)

        # r: road_id => ( (enter_lat, enter_lon), (leave_lat, leave_lon) )
        #road_ids_to_endpoints.update(r)
        #routes.append(r.keys())
        #gps_routes.append(gps_route)

    #return routes, gps_routes, road_ids_to_endpoints
    return {
            'routes': routes,
            'coordinate_routes': coordinate_routes,
            'departure_times': departure_times,
            'arrival_times': arrival_times,
            'road_ids_to_endpoints': road_ids_to_endpoints,
            }

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
    d = preprocess_data(sys.argv[1])

    routes = d['routes']
    road_ids_to_endpoints = d['road_ids_to_endpoints']
    coordinate_routes = np.array(d['coordinate_routes'])

    with Timer('sort road ids'):
        sorted_road_ids = np.array(sorted(road_ids_to_endpoints.keys()))

    #print("routes=", routes)
    #print("sorted_road_ids=", sorted_road_ids)

    with Timer('road ids to array'):
        a_road_ids = route_analysis.routes_to_array(routes, sorted_road_ids)

    with Timer('time features'):
        a_weekdays = np.zeros(shape=(len(routes), 7))
        a_hours = np.zeros(shape=(len(routes), 24))
        for i, departure_time in enumerate(d['departure_times']):
            try:
                dt = datetime.utcfromtimestamp(departure_time)
                a_weekdays[i, dt.weekday()] = 1.0
                a_hours[i, dt.hour] = 1.0
            except TypeError:
                pass

        non_roadid_features = 7 + 24

    #print(a_weekdays.shape, a_weekdays)
    #print(a_hours.shape, a_hours)
    #print(a_road_ids.shape, a_road_ids)
    #print(a_weekdays.shape, a_hours.shape, a_road_ids.shape)

    with Timer('hstack'):
        X = np.hstack((a_weekdays, a_hours, a_road_ids))



    with Timer('PCA'):
        pca = PCA(n_components=MAX_COMPONENTS)
        S_pca = pca.fit(X).transform(X)

    with Timer('plot PCA'):
        plots.all_relations(S_pca, '/tmp/pca.pdf')

    with Timer('ICA'):
        ica = FastICA(
                max_iter=4000,
                n_components=MAX_COMPONENTS,
                fun='exp',
                ) #max_iter=20000,
                #tol = 0.000001,
                #fun='cube',
                #n_components=3)
        S_ica = ica.fit(X).transform(X)  # Estimate the sources

    with Timer('plot ICA'):
        plots.all_relations(S_ica, '/tmp/ica.pdf')

    def render_road_ids(weights, name):
        trips = [road_ids_to_endpoints[id_] for id_ in sorted_road_ids]
        trip_weights = weights[non_roadid_features:]
        filename = '/tmp/gmaps_{}.html'.format(name)

        info = [
                gmaps.generate_html_bar_graph(weights[:7], ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']),
                gmaps.generate_html_bar_graph(weights[7:7+24], [str(i) for i in range(24)]),
                ]

        g = gmaps.generate_gmaps(
                center = trips[0][0],
                trips = trips,
                trip_weights = trip_weights,
                info = info)

        f = open(filename, 'w')
        f.write(g)
        f.close()

    def cluster_routes(X):
        metric = route_analysis.route_distance_jaccard
        #metric = route_analysis.route_distance_h1

        dbscan = DBSCAN(eps = 0.3, metric = metric).fit(X)
        labels = dbscan.labels_
        labels_unique = set(labels)


        for i in range(10):
            for j in range(10):
                print(i, j,
                        route_analysis.route_distance_jaccard(X[i,:], X[j,:]),
                        route_analysis.route_distance_h1(X[i,:], X[j,:]))

        colors = plt.cm.Spectral(np.linspace(0, 1, len(labels_unique)))

        print("labels=", labels_unique)

        for k, col in zip(labels_unique, colors):
            routes = X[labels == k]

            trips = []
            for route in routes:
                route = route[24 + 7:]
                trips.extend( [road_ids_to_endpoints[id_] for id_ in sorted_road_ids[route != 0]] )

            g = gmaps.generate_gmaps(
                    center = trips[0][0],
                    trips = trips,
                    default_color = '#ff00ff',
                    )

            f = open('/tmp/gmaps_dbscan_{}.html'.format(k + 1), 'w')
            f.write(g)
            f.close()

    with Timer('cluster'):
        cluster_routes(X)

    # Render mean

    with Timer('gmaps mean'):
        mean = np.average(X, axis=0)
        render_road_ids(mean, 'mean')

    # Render principal components

    with Timer('gmaps PCA'):
        components = pca.components_.T
        for i in range(min(MAX_COMPONENTS, components.shape[1])):
            component = components[:,i]
            render_road_ids(component, 'pc_{}'.format(i))

    # Render independent components

    with Timer('gmaps ICA'):
        components = ica.components_.T
        for i in range(min(MAX_COMPONENTS, components.shape[1])):
            component = components[:,i]
            render_road_ids(component, 'ic_{}'.format(i))

    print('\n'.join(Timer.pop_log()))


