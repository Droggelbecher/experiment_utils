#!/usr/bin/env python

import numpy as np
from sklearn.neighbors import KernelDensity
import scipy
import scipy.sparse
#import csv
import sys
from datetime import datetime
import math
from collections import Counter

from features import Features, Feature
import metrics

#csv.field_size_limit(sys.maxsize)

class Routes:
    def __init__(self, d):
        self.cv_range = (0, 0)

        #
        # Conversion between road ids, indices and coordinates
        #

        road_ids_to_endpoints = d['road_ids_to_endpoints']
        self.road_ids_to_endpoints = road_ids_to_endpoints

        # keys are of form (road_id, direction_flag)
        assert len(self.road_ids_to_endpoints.keys()[0]) == 2

        # values are of form ( (lat,lon), (lat,lon) )
        assert len(self.road_ids_to_endpoints.values()[0]) == 2
        assert len(self.road_ids_to_endpoints.values()[0][0]) == 2

        sorted_road_ids = np.array(sorted(road_ids_to_endpoints.keys()), dtype='int64, int8')
        self.id_to_idx = { tuple(id_): idx for (idx, id_) in enumerate(sorted_road_ids) }
        #self.startpoints = np.array([road_ids_to_endpoints[tuple(id_)][0] for id_ in sorted_road_ids])
        #self.endpoints = np.array([road_ids_to_endpoints[tuple(id_)][1] for id_ in sorted_road_ids])
        self.idx_to_endpoints = np.array([road_ids_to_endpoints[tuple(id_)] for id_ in sorted_road_ids])

        #
        # Routes
        #

        routes = d['routes']
        self._routes = routes

        coordinate_routes = d['coordinate_routes']
        self._coordinate_routes = coordinate_routes

        assert len(self._routes) == len(self._coordinate_routes)

        self.F = Features(
                Feature('weekdays',     np.arange(0, 7, 1.0),  'chist_wrap', weight = 0.0),
                Feature('hours',        np.arange(0, 24, 1.0), 'chist_wrap', weight = 0.0),
                Feature('arrival',      ('lat', 'lon'),        'geo', weight = 0.00),
                #Feature('departure',    ('lat', 'lon'),        'geo', weight = 0.00),
                Feature('route',        sorted_road_ids,       'set', weight = 1.0),
                #Feature('arrival_arcs', sorted_road_ids,       'set', weight = 0.0),
                )

        a_arrival = np.array([c[-1] for c in coordinate_routes])
        a_departure = np.array([c[0] for c in coordinate_routes])

        # Departure Time
        #
        a_weekdays = np.zeros(shape=(len(routes), 7))
        a_hours = np.zeros(shape=(len(routes), len(self.F.hours)))
        for i, departure_time in enumerate(d['departure_times']):
            try:
                dt = datetime.utcfromtimestamp(departure_time)
                a_weekdays[i, dt.weekday()] = 1.0

                # Flag a '1' for all hour-categories that match.
                # Each category is a centered range of length 1hr around each
                # key, if eg. the resolution is 30min, there is always 2
                # overlapping

                for j, key in enumerate(self.F.hours.keys):
                    if abs(key - dt.hour) < 1.0:
                        a_hours[i, j] = 1.0

            except TypeError:
                pass

        #
        # guess partial adjacency matrix from data
        #
        A = np.zeros(shape=(len(road_ids_to_endpoints), len(road_ids_to_endpoints)))
        idx = self.id_to_idx
        for route in routes:
            for id1, id2 in zip(route, route[1:]):
                A[idx[id1], idx[id2]] = 1.0

        self.A = A

        a_road_ids, a_arrival_arcs = routes_to_array(routes, self.id_to_idx)

        # TODO: this needs to stay consistent with the offsets above
        self._X = np.hstack((
            a_weekdays,
            a_hours,
            a_arrival,
            #a_departure,
            a_road_ids,
            #a_arrival_arcs
            ))

        #self._feature_density = KernelDensity(kernel='gaussian',
                #bandwidth=1.0).fit(self.get_features(self._X))

    def _validation_index_to_abs(self, i):
        return self.cv_range[0] + i

    def _learn_index_to_abs(self, i):
        return i if i < self.cv_range[0] else self.cv_range[1] + i

    def _learn_range(self):
        for i in range(self.cv_range[0]):
            yield i
        for i in range(self.cv_range[1], self._X.shape[0]):
            yield i

    def _validation_range(self):
        for i in range(self.cv_range[0], self.cv_range[1]):
            yield i


    def get_learn_X(self):
        a = np.vstack((self._X[:self.cv_range[0],:], self._X[self.cv_range[1]:,:]))
        assert a.shape[1] == self._X.shape[1]
        assert a.shape[0] == self._X.shape[0] - (self.cv_range[1] - self.cv_range[0])
        return a

    def get_validation_X(self):
        return self._X[self.cv_range[0], self.cv_range[1]]


    def get_learn_coordinate_routes(self):
        for i in self._learn_range():
            yield self._coordinate_routes[i]

    def get_learn_routes(self):
        for i in self._learn_range():
            yield self._routes[i]

    def get_validation_routes(self):
        for i in self._validation_range():
            yield self._routes[i]


    def get_learn_features(self):
        for i in self._learn_range():
            yield self.F.all_except('route', self._X[i])

    def get_validation_features(self):
        for i in self._validation_range():
            yield self.F.all_except('route', self._X[i])



    def get_features(self, a, features = set()):
        if not features:
            features = self.F.get_names() - set(['route'])
        if isinstance(a, int):
            a = self._X[a]
        return self.F.extract(a, features)

    def route_to_array(self, route, default = 0.0, features = None):
        """
        Turn a list of road ids into a data row compatible with self.X
        """
        route, arrival, unknowns = routes_to_array([route], self.id_to_idx, unknowns = 'count', default = default)
        return self.F.assemble(route = route, _rest = features), unknowns

    def array_to_route(self, r):
        """
        Turn array into a set of road ids (that is without order)

        >>> r = list(range(100))
        >>> r2 = array_to_route( route_to_array(r) )
        >>> set(r) == set(r)
        True
        """
        return self.F.route.keys[self.F.route(r) != 0]

    def array_to_point_pairs(self, r):
        route = self.F.route(r)
        point_pairs = [(tuple(sx), tuple(ex)) for sx, ex in self.idx_to_endpoints[route != 0, :]]
        return point_pairs

    def distance(self, r1, r2, **kws):
        return self.F.distance(r1, r2, **kws)

    def nearest_neighbor(self, r, **kws):
        d_min = float('inf')
        r_min = None

        for r2 in self.get_learn_X():
            d = self.distance(r, r2, **kws)
            if d < d_min:
                d_min = d
                r_min = r2

        return r_min, d_min


def to_directed_arcs(route, coordinate_route, road_ids_to_endpoints):
    """
    routes: iterable over road ids
    coordinate_routes: iterable over (lat, lon) pairs describing entry points to those roads
    road_ids_to_endpoint: dict { road_id: ((lat_enter, lon_enter), (lat_leave, lon_leave)) }

    return: iterable over (road_id, direction) pairs where direction is either 0 or 1.
        Last road_id is discarded (as only one point of that arc is known)
    """

    for road_id, coord in zip(route, coordinate_route):
        try:
            enter, leave = road_ids_to_endpoints[road_id]
        except KeyError:
            # There are, alas no differentiable entry- and leave points
            # for this arc

            print("road_id {} not directable!".format(road_id))
            continue

        dist_enter = metrics.geo(coord, enter)
        dist_leave = metrics.geo(coord, leave)

        assert enter != leave

        if dist_enter > dist_leave:
            yield (road_id, 0)
        else:
            yield (road_id, 1)

def find_cycle(route):

    for i, e in enumerate(route):
        try:
            idx = route[:i].index(e)
            return idx, i, e
        except ValueError, e:
            pass

    return None

def remove_duplicates(route, *related):
    seen = set()
    seen_add = seen.add
    #return [x for x in route if not (x in seen or seen_add(x))]

    z = zip(route, *related)
    return zip(*[x for x in z if not (x[0] in seen or seen_add(x))])


def routes_to_array(routes, ids, unknowns = 'raise', default = 0.0):
    """
    routes: iterable over (iterable over route ids)
    ids: dict: id->idx

    return: sparse matrix with shape (len(routes), len(ids)),
        containing 1 if road id belongs to that route
    """
    assert type(ids) is dict

    a = np.full((len(routes), len(ids)), default)
    a_arrival = np.full((len(routes), len(ids)), default)

    u = 0

    for i, route in enumerate(routes):
        for r in route:
            try:
                rid = ids[r]
            except IndexError as e:
                u += 1
                if unknowns == 'raise':
                    raise
            else:
                a[i, rid] = 1

        if len(route) and (unknowns == 'raise' or route[-1] in ids):
            a_arrival[i, ids[route[-1]]] = 1

    if unknowns == 'raise':
        return a, a_arrival
    else:
        return a, a_arrival, u




def roadid_covariance_matrix(routes):
    """
    routes: iterable over (iterarable over road ids)

    return: tuple (ids, means, covariance)
        ids: array of all possible road ids, sorted
        means: array of means (1d, 1 mean for each road-id in order)
        covariance: covariance matrix len(ids)*len(ids)

        len(ids) == len(means)
    """
    all_road_ids = set()
    for route in routes:
        all_road_ids.update(route)
    all_road_ids = np.array(sorted(all_road_ids))


    co_occurence = Counter()
    for route_a in routes:
        for road_id_i in route_a:
            for road_id_j in route_a:
                if road_id_i <= road_id_j:
                    co_occurence[road_id_i, road_id_j] += 1



    covariance = np.zeros(shape = (len(all_road_ids), len(all_road_ids)), dtype=np.float_)

    L = float(len(routes))
    LL = L ** 2.0
    means = np.zeros(shape=(len(all_road_ids)))

    for i, road_id_i in enumerate(all_road_ids):
        means[i] = co_occurence[road_id_i, road_id_i] / L

        for j, road_id_j in enumerate(all_road_ids[i:]):
            c = co_occurence[road_id_i, road_id_j] / L
            c -= co_occurence[road_id_i, road_id_i] * co_occurence[road_id_j, road_id_j] / LL

            covariance[i, i + j] = c
            covariance[i + j, i] = c

    return (all_road_ids, means, covariance)

if __name__ == '__main__':
    ids, cov = roadid_covariance_matrix([sys.argv[1]])
    print cov
    print np.where(cov != 0)



