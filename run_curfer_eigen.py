#!/usr/bin/env python

import sys
import os
import os.path
import shutil
import subprocess
import math
import itertools
from datetime import datetime
import random

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

sys.path.append('sklearn_autoencoder')
from autoencoder import DenoisingAutoencoder

from cache import cached, NEVER, ALWAYS
from navkit import prepare_positioning, prepare_mapmatching, run_positioning, run_mapmatching
from route_model_simmons import RouteModelSimmons, RouteException
from route_model_simmons_pca import RouteModelSimmonsPCA
from route_model_simmons_nofeatures import RouteModelSimmonsNoFeatures
from timer import Timer
import curfer
import geo
import gmaps
import osutil
import plots
import route_analysis
import ttp
import iterutils
import util
from util import C

np.set_printoptions(threshold=9999999999,linewidth=99999,precision=3)


GPS_TTP_FILENAME = '/tmp/gps.ttp'
MAX_COMPONENTS = 3
HAVE_NAVKIT = False
RENDER_COMPONENTS = False


@cached(filename_kws = ['curfer_filename'], compute_if=ALWAYS if HAVE_NAVKIT else NEVER)
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

    if HAVE_NAVKIT:
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

        for k, (a, b) in d['road_ids_to_endpoints'].items():
            # Only update if new entry is unambiguous
            if a != b:
                road_ids_to_endpoints[k] = (a, b)
        #road_ids_to_endpoints.update(d['road_ids_to_endpoints'])
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
    filename = '/tmp/gmaps_{}.html'.format(name)

    route = r.F.route(weights)

    info = [
            'Route min: {} max: {}'.format(min(route), max(route)),
            gmaps.generate_html_bar_graph(r.F.weekdays(weights), r.F.weekdays.keys),
            gmaps.generate_html_bar_graph(r.F.hours(weights), r.F.hours.keys),
            ]

    lines = itertools.chain(
                gmaps.weighted_lines(route, r.endpoints),
                gmaps.weighted_lines(r.F.arrival_arcs(weights), r.endpoints, '#ff00ff', '#000000', opacity = 1.0)
                )

    g = gmaps.generate_gmaps(
            center = r.endpoints[0][0],
            lines = lines,
            info = info)

    f = open(filename, 'w')
    f.write(g)
    f.close()


def cluster_routes(r):
    metric = r.distance_jaccard

    print('r.X=', r.X.shape)
    dbscan = DBSCAN(eps = 0.5, metric = metric).fit(r.X)
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

        line_sets = gmaps.line_sets((r.endpoints[r.F.route(route) != 0] for route in routes))


        #colors = plt.cm.Spectral(np.linspace(0, 1, len(routes)))

        weights = np.zeros(r.X.shape[1])

        for route in routes:
            weights += route

        weights /= len(routes)

        arrival_radius = max(geo.distance(r.F.arrival(row), r.F.arrival(weights)) for row in routes)
        departure_radius = max(geo.distance(r.F.departure(row), r.F.departure(weights)) for row in routes)

        g = gmaps.generate_gmaps(
                center = r.endpoints[0][0],
                lines = line_sets,
                markers = [ r.F.arrival(weights) ],
                circles = [
                    {
                        'center': { 'lat': weights[r.F.arrival.start], 'lng': weights[r.F.arrival.start+1] },
                        'radius': arrival_radius,
                        'strokeColor': '#ffffff',
                    },
                    {
                        'center': { 'lat': weights[r.F.departure.start], 'lng': weights[r.F.departure.start+1] },
                        'radius': departure_radius,
                        'strokeColor': '#000000',
                    }],
                info = [
                    '# routes: {}'.format(len(routes)),
                    'departure radius (black): {}'.format(departure_radius),
                    'arrival radius (white): {}'.format(arrival_radius),
                    gmaps.generate_html_bar_graph(r.F.weekdays(weights), r.F.weekdays.keys),
                    gmaps.generate_html_bar_graph(r.F.hours(weights),    r.F.hours.keys),
                    ]
                )

        f = open('/tmp/gmaps_dbscan_{}.html'.format(k + 1), 'w')
        f.write(g)
        f.close()


def analyze(r):
    with Timer('PCA'):
        pca = PCA(n_components=MAX_COMPONENTS)
        S_pca = pca.fit(r.X).transform(r.X)

    with Timer('plot PCA'):
        plots.all_relations(S_pca, '/tmp/pca.pdf')

    with Timer('ICA'):
        ica = FastICA(
                max_iter=32000,
                n_components=MAX_COMPONENTS,
                #fun='exp',
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




def jaccard(r1, r2):
    s1 = set(r1)
    s2 = set(r2)
    union = s1.union(s2)
    if len(union) == 0:
        return 0
    return len(s1.intersection(s2)) / float(len(union))

def eval_metric(predicted, expected):
    IGNORE_LAST_ARCS = 5
    return jaccard(predicted[:-IGNORE_LAST_ARCS], expected[:-IGNORE_LAST_ARCS])

def test_predict_route(model, partial, expected, features, stats):
    stats['length_partial'].append(len(partial))
    stats['length_expected'].append(len(expected))

    try:
        predicted, likelihood = model.predict_route(partial, features)

    except RouteException as e:
        predicted = e.route
        likelihood = 0
        stats['exception'].append(1)

    else:
        stats['exception'].append(0)

    score = eval_metric(predicted, expected)

    stats['likelihood'].append(likelihood)
    stats['score'].append(score)
    stats['length_predicted'].append(len(predicted))

    return predicted






def test_partial_prediction(d):
    road_ids_to_endpoints = d['road_ids_to_endpoints']


    def plot_gmaps(partial, continuation, name, info):
        """
        Plot on a gmap a partial route and its continuation.
        """
        lines = gmaps.line_sets([
            [road_ids_to_endpoints[x[0]] for x in partial],
            [road_ids_to_endpoints[x[0]] for x in continuation],
            ])
        #info = ['{}: {}'.format(k, v) for k, v in kws.items()]
        g = gmaps.generate_gmaps(center = road_ids_to_endpoints.values()[0][0], lines = lines,
                info = info)
        f = open('/tmp/gmaps_{}.html'.format(name), 'w')
        f.write(g)
        f.close()


    with Timer("preparation"):

        # Preprocess routes:
        # 1. assign directions
        # 2. remove duplicates (so routes contain no cycles)
        routes = [
                    list(route_analysis.remove_duplicates(route_analysis.to_directed_arcs(route, coordinate_route, road_ids_to_endpoints)))
                for route, coordinate_route, departure_time in
                zip(d['routes'], d['coordinate_routes'], d['departure_times'])
                ]

        # Compute feature vectors (weekday & time of day)

        features = [] 
        for i, departure_time in enumerate(d['departure_times']):
            f = np.zeros(7 + 24)
            if departure_time is not None:
                dt = datetime.utcfromtimestamp(departure_time)
                f[dt.weekday()] = 1.0
                f[7 + dt.hour] = 1.0
            features.append(f)

        print("total routes:", len(routes))

        ziproutes = zip(routes, features)
        random.shuffle(ziproutes, lambda: 0.42)


        # Cross-validate prediction accuracy
        #
        CV_FACTOR = 10
        chunks = list(iterutils.chunks(ziproutes, CV_FACTOR))


        results = {}

    #for partial_length in (0.0, 0.25, 0.5, 0.75):
    for partial_length in (0.0, 0.1, 0.2, 0.3):
    #for partial_length in (0.1,):
        route_models = [
                C(name = 'SimmonsNoF',  make = RouteModelSimmonsNoFeatures,     stats = util.listdict()),
                #C(name = 'Simmons',     make = RouteModelSimmons,               stats = util.listdict()),
                C(name = 'SimmonsPCA1', make = lambda: RouteModelSimmonsPCA(PCA(n_components = 1)), stats = util.listdict()),
                #C(name = 'SimmonsPCA2', make = lambda: RouteModelSimmonsPCA(PCA(n_components = 2)), stats = util.listdict()),
                #C(name = 'SimmonsPCA3', make = lambda: RouteModelSimmonsPCA(PCA(n_components = 3)), stats = util.listdict()),

                # Is great (that is, slightly better than PCA, exactly as good
                # as NoF [ :( ], costs a long time to compute (~5-7min per CV)
                #C(name = 'SimmonsAutoEnc10', make = lambda: RouteModelSimmonsPCA(DenoisingAutoencoder(10)), stats = util.listdict()),
                ]
        results[partial_length] = route_models

        for cv_idx in range(CV_FACTOR):
            print("cv_idx={} routes={}".format(cv_idx, len(chunks[cv_idx])))

            for d in route_models:

                with Timer('{} {}/{}'.format(d.name, cv_idx, CV_FACTOR)):

                    # train on all chunks but cv_idx
                    
                    model = d.make()
                    routes = list(itertools.chain(*(chunks[:cv_idx] + chunks[cv_idx + 1:])))
                    model.learn_routes(routes, road_ids_to_endpoints)

                    # If model has components, render them

                    if RENDER_COMPONENTS and hasattr(model, '_decompositor'):
                        if hasattr(model._decompositor, 'components_'):
                            sorted_road_ids = np.array(sorted(road_ids_to_endpoints.keys()))
                            endpoints = np.array([road_ids_to_endpoints[id_] for id_ in sorted_road_ids])

                            components = model._decompositor.components_.T
                            for i in range(components.shape[1]):
                                component = components[:,i]
                                route = component[7 + 24:]
                                info = [
                                        'Route min: {} max: {}'.format(min(route), max(route)),
                                        gmaps.generate_html_bar_graph(component[:7], range(7)),
                                        gmaps.generate_html_bar_graph(component[7:7+24], range(24)),
                                        ]
                                lines = gmaps.weighted_lines(route, endpoints)
                                g = gmaps.generate_gmaps(center =
                                        endpoints[0][0], lines =
                                        lines, info = info)
                                f = open('/tmp/{}_{}_{}_comp_{}.html'.format(d.name, partial_length, cv_idx, i), 'w')
                                f.write(g)
                                f.close()

                    # evaluate on cv_idx

                    for i, (route, features) in enumerate(chunks[cv_idx]):

                        l = max(1, int(partial_length * len(route)))
                        expected = route[l:]

                        # As a ground for comparison find out the best possible
                        # score the test route can achieve (that is max similarity
                        # to any known route):
                        s_max = 0
                        route_max = None
                        f_max = None
                        for (route2, f2) in routes:
                            s = eval_metric(expected, route2[l:])
                            if s > s_max:
                                s_max = s
                                route_max = route2
                                f_max = f2
                                
                                # Ok, we got it, we should be able to predict this
                                # one pretty well
                                if s_max >= .99:
                                    break

                        if s_max < 0.8:
                            continue

                        d.stats['test_route_score'].append(s_max)

                        partial = route[:l]
                        predicted = test_predict_route(model, partial, expected, features, d.stats)

                        print("{} cv {} i {} likelihood {} score {} partial/rel {} partial/abs {} explen {} predlen {}".format(
                            d.name, cv_idx, i,
                            d.stats['likelihood'][-1],
                            d.stats['score'][-1],
                            partial_length,
                            l,
                            len(expected),
                            len(predicted)))

                        plot_gmaps(partial, expected, '{}_{}_{}_{}_expected'.format(d.name, partial_length, cv_idx, i),
                                info = [
                                    gmaps.generate_html_bar_graph(features[:7], 'Mo Tu We Th Fr Sa Su'.split()),
                                    gmaps.generate_html_bar_graph(features[7:7+24], range(24)),
                                    ])

                        plot_gmaps(partial, route_max, '{}_{}_{}_{}_best'.format(d.name, partial_length, cv_idx, i),
                                info = [
                                    gmaps.generate_html_bar_graph(f_max[:7], 'Mo Tu We Th Fr Sa Su'.split()),
                                    gmaps.generate_html_bar_graph(f_max[7:7+24], range(24)),
                                    ])

                        plot_gmaps(partial, predicted, '{}_{}_{}_{}_predicted'.format(d.name, partial_length, cv_idx, i),
                                info = [
                                    '{}: {}'.format(k, v[-1]) for k,v in d.stats.items()
                                    #'{}: {}'.format((k, v[-1] if len(v) else '-') for k, v in d.stats.items())
                                    ])

    with Timer("plotting"):
        
        for partial_length, route_models in results.items():
            for d in route_models:
                scores = d.stats['score']

                print("total partial length {:5.4f} {:20s} score/jaccard min {:5.4f} avg {:5.4f} max {:5.4f}".format(
                    partial_length,
                    d.name,
                    min(scores), sum(scores)/len(scores), max(scores)))

                plots.relation(d.stats['test_route_score'], d.stats['score'], '/tmp/{}_{}_rel_score.pdf'.format(d.name, partial_length))
        

        ls = sorted(results.keys())

        plots.multi_boxplots(
                xs = ls,
                ysss = [ [results[l][i].stats['score'] for l in ls] for i in range(len(route_models)) ],
                labels = [rm.name for rm in route_models],
                ylim = (-.05, 1.05),
                filename = '/tmp/scores_by_length.pdf'
                )


        #
        # Group things by Model (and nothing else):
        #

        for i in range(len(results.values()[0])):

            a = np.array([
                    list(itertools.chain(*[results[l][i].stats['length_partial'] for l in ls])),
                    list(itertools.chain(*[results[l][i].stats['length_expected'] for l in ls])),
                    list(itertools.chain(*[results[l][i].stats['length_predicted'] for l in ls])),
                    list(itertools.chain(*[results[l][i].stats['likelihood'] for l in ls])),
                    list(itertools.chain(*[results[l][i].stats['score'] for l in ls])),
                    #list(itertools.chain(*[results[l][i].stats['exception'] for l in ls])),
                    list(itertools.chain(*[results[l][i].stats['test_route_score'] for l in ls])),
                    ]).T

            plots.all_relations(a, '/tmp/{}_stats.pdf'.format(route_models[i].name),
                labels = ['l(part)', 'l(exp)', 'l(pred)', 'likely', 'score', 'route score'])

if __name__ == '__main__':
    d = preprocess_data(sys.argv[1])

    test_partial_prediction(d)

    #r = route_analysis.Routes(d)
    #analyze(r)


    print('\n'.join(Timer.pop_log()))


