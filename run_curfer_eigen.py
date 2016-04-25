#!/usr/bin/env python

import sys
import os
import os.path
import shutil
import subprocess
import math
import itertools

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

from cache import cached, NEVER, ALWAYS
from navkit import prepare_positioning, prepare_mapmatching, run_positioning, run_mapmatching
from route_model_simmons import RouteModelSimmons, CyclicRouteException
from route_model_simmons_pca import RouteModelSimmonsPCA
from timer import Timer
import curfer
import geo
import gmaps
import osutil
import plots
import route_analysis
import ttp
import iterutils
from util import C

np.set_printoptions(threshold=9999999999,linewidth=99999,precision=3)


GPS_TTP_FILENAME = '/tmp/gps.ttp'
MAX_COMPONENTS = 10
HAVE_NAVKIT = False


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

    info = [
            gmaps.generate_html_bar_graph(r.F.weekdays(weights), r.F.weekdays.keys),
            gmaps.generate_html_bar_graph(r.F.hours(weights), r.F.hours.keys),
            ]

    lines = itertools.chain(
                gmaps.weighted_lines(r.F.route(weights), r.endpoints),
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
            #route = r.F.route(route)
            #ep = r.endpoints[route != 0]
            #trips.extend(ep)
            #trip_colors.extend([rgb2hex(c)] * len(ep))

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



def test_partial_prediction(d):
    road_ids_to_endpoints = d['road_ids_to_endpoints']


    def plot_gmaps(partial, continuation, name, **kws):
        lines = gmaps.line_sets([
            [road_ids_to_endpoints[x[0]] for x in partial],
            [road_ids_to_endpoints[x[0]] for x in continuation],
            ])
        info = ['{}: {}'.format(k, v) for k, v in kws.items()]
        g = gmaps.generate_gmaps(center = road_ids_to_endpoints.values()[0][0], lines = lines,
                info = info)
        f = open('/tmp/gmaps_{}.html'.format(name), 'w')
        f.write(g)
        f.close()


    # Preprocess routes:
    # 1. assign directions
    # 2. remove duplicates (so routes contain no cycles)
    routes = [
            list(route_analysis.remove_duplicates(route_analysis.to_directed_arcs(route, coordinate_route, road_ids_to_endpoints)))
            for route, coordinate_route in zip(d['routes'], d['coordinate_routes'])
            ]

    print("total routes:", len(routes))


    # Cross-validate prediction accuracy
    #
    CV_FACTOR = 10
    chunks = list(iterutils.chunks(routes, CV_FACTOR))

    partial_length = 0.25


    def jaccard(r1, r2):
        s1 = set(r1)
        s2 = set(r2)
        return len(s1.intersection(s2)) / float(len(s1.union(s2)))

    def eval_metric(predicted, expected):
        return jaccard(predicted[:-5], expected[:-5])


    route_models = [
            C(name = 'Simmons', class_ = RouteModelSimmons,
                scores = [], likelihoods = []),
            C(name = 'SimmonsPCA', class_ = RouteModelSimmonsPCA,
                scores = [], likelihoods = [])
            ]

    for cv_idx in range(CV_FACTOR):
        print("cv_idx={} routes={}".format(cv_idx, len(chunks[cv_idx])))

        for d in route_models:
            model = d.class_()
            scores = d.scores
            likelihoods = d.likelihoods
            name = d.name

            model.learn_routes(
                    list(itertools.chain(*(chunks[:cv_idx] + chunks[cv_idx + 1:]))),
                    road_ids_to_endpoints
                    )

            for i, route in enumerate(chunks[cv_idx]):
                l = int(partial_length * len(route))
                partial = route[:l]
                expected = route[l:]

                try:
                    predicted, likelihood = model.predict_route(partial)
                    score = eval_metric(predicted, expected)
                    scores.append(score)
                    likelihoods.append(likelihood)

                except CyclicRouteException as e:
                    score = -1
                    likelihood = -1
                    print(e)
                    print('route=', e.route)
                    sys.stdout.flush()
                    #predicted = e.route
                    #plot_gmaps(partial, expected, 'cycle_expected')
                    #plot_gmaps(partial, predicted, 'cycle_predicted')


                print("cv {} i {} likelihood {} score {}".format(cv_idx, i, likelihood, score))
                plot_gmaps(partial, expected,
                        '{}_{}_{}_expected'.format(name, cv_idx, i))
                plot_gmaps(partial, predicted,
                        '{}_{}_{}_predicted'.format(name, cv_idx, i),
                        likelihood = likelihood,
                        score = score)


    for d in route_models:
        scores = d.scores
        likelihoods = d.likelihoods
        name = d.name

        print("total {:20s} score/jaccard min {:5.4f} avg {:5.4f} max {:5.4f}".format(
            name,
            min(scores), sum(scores)/len(scores), max(scores)))

        plots.relation(likelihoods, scores, '/tmp/{}_likelihood_score.pdf'.format(name))


    plots.cdfs([dict(label = d.name, values = d.scores) for d in route_models], '/tmp/scores.pdf')


if __name__ == '__main__':
    d = preprocess_data(sys.argv[1])

    #train_simmons(d)
    test_partial_prediction(d)

    # r = route_analysis.Routes(d)
    # analyze(r)


    print('\n'.join(Timer.pop_log()))


