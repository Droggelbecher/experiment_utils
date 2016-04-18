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
import geo

np.set_printoptions(threshold=99999,linewidth=99999,precision=3)


GPS_TTP_FILENAME = '/tmp/gps.ttp'
MAX_COMPONENTS = 10



class Routes:
    def __init__(self, d):
        routes = d['routes']
        road_ids_to_endpoints = d['road_ids_to_endpoints']
        coordinate_routes = d['coordinate_routes']
        sorted_road_ids = np.array(sorted(road_ids_to_endpoints.keys()))
        self.endpoints = np.array([road_ids_to_endpoints[id_] for id_ in sorted_road_ids])

        a_arrival = np.array([c[-1] for c in coordinate_routes])
        a_departure = np.array([c[0] for c in coordinate_routes])

        # Positions of stuff in X (columns=features)

        p = 0
        self.weekdays_start = p; p += 7;
        self.weekdays_end = p
        self.hours_start = p; p += 24
        self.hours_end = p
        self.arrival_start = p; p+= 2
        self.arrival_end = p

        self.routes_start = p

        # Departure Time

        a_weekdays = np.zeros(shape=(len(routes), 7))
        a_hours = np.zeros(shape=(len(routes), 24))
        for i, departure_time in enumerate(d['departure_times']):
            try:
                dt = datetime.utcfromtimestamp(departure_time)
                a_weekdays[i, dt.weekday()] = 1.0
                a_hours[i, dt.hour] = 1.0
            except TypeError:
                pass

        self.sorted_road_ids = sorted_road_ids
        a_road_ids = route_analysis.routes_to_array(routes, sorted_road_ids)
        self.X = np.hstack((
            a_weekdays,
            a_hours,
            a_arrival,
            a_road_ids
            ))

    def get_endpoints(self):
        return self.endpoints

    #def get_arrivals(self):
        #return self.arrivals

    #def get_departures(self):
        #return self.departures

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

    return {
            'routes': routes,
            'coordinate_routes': coordinate_routes,
            'departure_times': departure_times,
            'arrival_times': arrival_times,
            'road_ids_to_endpoints': road_ids_to_endpoints,
            }

def render_road_ids(r, weights, name):
    trips = r.get_endpoints()
    trip_weights = weights[r.routes_start:]
    filename = '/tmp/gmaps_{}.html'.format(name)

    info = [
            gmaps.generate_html_bar_graph(weights[:r.weekdays_end], ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']),
            gmaps.generate_html_bar_graph(weights[r.hours_start:r.hours_end], [str(i) for i in range(24)]),
            ]

    g = gmaps.generate_gmaps(
            center = trips[0][0],
            trips = trips,
            trip_weights = trip_weights,
            info = info)

    f = open(filename, 'w')
    f.write(g)
    f.close()


def cluster_routes(r):
    metric = route_analysis.route_distance_jaccard
    #metric = route_analysis.route_distance_h1

    dbscan = DBSCAN(eps = 0.3, metric = metric).fit(r.X)
    labels = dbscan.labels_
    labels_unique = set(labels)

    # DEBUG: print some distances to check metric
    #for i in range(10):
        #for j in range(10):
            #print(i, j,
                    #route_analysis.route_distance_jaccard(r.X[i,:], r.X[j,:]),
                    #route_analysis.route_distance_h1(r.X[i,:], r.X[j,:]))

    print("DBSCAN labels = {}".format(' '.join(str(x) for x in labels_unique)))

    for k in labels_unique:
        routes = r.X[labels == k]

        trips = []
        trip_colors = []
        colors = plt.cm.Spectral(np.linspace(0, 1, len(routes)))

        weights = np.zeros(r.X.shape[1])

        for c,route in zip(colors, routes):
            weights += route
            route = route[r.routes_start:]
            ep = r.endpoints[route != 0]
            trips.extend(ep)
            trip_colors.extend([rgb2hex(c)] * len(ep))

        weights /= len(routes)

        radius = max(geo.distance(row[r.arrival_start], row[r.arrival_start + 1],
            weights[r.arrival_start], weights[r.arrival_start + 1]) for row in routes)

        g = gmaps.generate_gmaps(
                center = trips[0][0],
                trips = trips,
                trip_colors = trip_colors,
                default_color = '#ff00ff',
                markers = [ weights[r.arrival_start:r.arrival_start+2] ],
                circles = [ (weights[r.arrival_start], weights[r.arrival_start+1], radius) ],
                info = [
                    'routes: {}'.format(len(routes)),
                    gmaps.generate_html_bar_graph(weights[:r.weekdays_end], ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']),
                    gmaps.generate_html_bar_graph(weights[r.hours_start:r.hours_end], [str(i) for i in range(24)]),
                    ]
                )

        f = open('/tmp/gmaps_dbscan_{}.html'.format(k + 1), 'w')
        f.write(g)
        f.close()



if __name__ == '__main__':
    d = preprocess_data(sys.argv[1])

    r = Routes(d)

    with Timer('PCA'):
        pca = PCA(n_components=MAX_COMPONENTS)
        S_pca = pca.fit(r.X).transform(r.X)

    with Timer('plot PCA'):
        plots.all_relations(S_pca, '/tmp/pca.pdf')

    with Timer('ICA'):
        ica = FastICA(
                max_iter=4000,
                n_components=MAX_COMPONENTS,
                fun='exp',
                )
        S_ica = ica.fit(r.X).transform(r.X)  # Estimate the sources

    with Timer('plot ICA'):
        plots.all_relations(S_ica, '/tmp/ica.pdf')

    with Timer('cluster'):
        cluster_routes(r)

    # Render mean

    with Timer('gmaps mean'):
        mean = np.average(r.X, axis=0)
        render_road_ids(r, mean, 'mean')

    # Render principal components

    with Timer('gmaps PCA'):
        components = pca.components_.T
        for i in range(min(MAX_COMPONENTS, components.shape[1])):
            component = components[:,i]
            render_road_ids(r, component, 'pc_{}'.format(i))

    # Render independent components

    with Timer('gmaps ICA'):
        components = ica.components_.T
        for i in range(min(MAX_COMPONENTS, components.shape[1])):
            component = components[:,i]
            render_road_ids(r, component, 'ic_{}'.format(i))

    print('\n'.join(Timer.pop_log()))


