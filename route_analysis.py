#!/usr/bin/env python

import numpy as np
from sklearn.neighbors import KernelDensity
import scipy
import scipy.sparse
import sys
from datetime import datetime
import math
from collections import Counter, defaultdict
import logging

from features import Features, PlainFeature, GeoFeature, BitSetFeature, OneHotFeature
import metrics
import plots

#csv.field_size_limit(sys.maxsize)

class Routes:
    def __init__(self, track_reader):
        self.cv_range = (0, 0)

        ids = set()
        len_tracks = 0
        for track in track_reader():
            len_tracks += 1
            for arc in track['track']:
                ids.add(arc['roadid'])

        ids.remove(None)
        sorted_road_ids = sorted(ids) + [None]

        id_to_idx = { v: i for i, v in enumerate(sorted_road_ids) }

        self.F = Features(
                GeoFeature('origin'),
                GeoFeature('destination'),
                PlainFeature('day_of_week', (0, 7)),
                PlainFeature('hour_of_day', (0, 24)),
                # Minutes since midnight
                PlainFeature('minute_of_day', (0, 24 * 60)),
                # Keeping  the one-hot variants now for data display in html
                OneHotFeature('day_of_week_onehot', range(7)),
                OneHotFeature('hour_of_day_onehot', range(24)),
                BitSetFeature('route', sorted_road_ids),
                )

        X = np.zeros(shape = (len_tracks, len(self.F)))

        id_to_point_pair = {}
        for i, track in enumerate(track_reader()):
            id_to_point_pair = merge_arc_coordinates( id_to_point_pair, extract_arc_coordinates(track) )

            if i % 10 == 0:
                logging.debug('{:4d}/{:4d} tracks read'.format(i, len_tracks))

            origin = track['track'][0]
            destination = track['track'][-1]

            self.F.origin.encode(X[i, :], (origin['lat'], origin['lon']))
            self.F.destination.encode(X[i, :], (destination['lat'], destination['lon']))

            # Do we have a departure time?
            deptime = origin.get('t', None)

            if deptime is not None:
                deptime = float(deptime)
                dt = datetime.utcfromtimestamp(deptime)
                self.F.day_of_week.encode(X[i, :], dt.weekday())
                self.F.hour_of_day.encode(X[i, :], dt.hour)
                self.F.minute_of_day.encode(X[i, :], dt.hour * 60 + dt.minute)
                self.F.day_of_week_onehot.encode(X[i, :], dt.weekday())
                self.F.hour_of_day_onehot.encode(X[i, :], dt.hour)

            ids = set()
            for ris in track['roadid_seq']:
                ids.add(id_to_idx[ris[0]])
                a_ids = np.zeros(len(sorted_road_ids))
                for id in ids:
                    a_ids[id] = 1.0
                self.F.route.encode(X, a_ids)

        id_to_point_pair2 = dict(id_to_point_pair)
        id_to_point_pair2[None] = ((0.0, 0.0), (0.0, 0.0))

        self.id_to_point_pair = id_to_point_pair
        self.len_tracks = len_tracks
        self.id_to_idx = id_to_idx
        self.idx_to_id = sorted_road_ids
        self.idx_to_point_pairs = [id_to_point_pair2[x] for x in sorted_road_ids]
        #self._routes = tracks
        self._X = X
        self.track_reader = track_reader


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


    def get_learn_tracks(self):
        for i, t in enumerate(self.track_reader()):
            if i < self.cv_range[0] or i >= self.cv_range[1]:
                yield [x[0] for x in t['roadid_seq']]

    def get_validation_tracks(self):
        for i, t in enumerate(self.track_reader()):
            if i < self.cv_range[0] or i >= self.cv_range[1]:
                yield [x[0] for x in t['roadid_seq']]

    def get_validation_routenames(self):
        for i, t in enumerate(self.track_reader()):
            if i < self.cv_range[0] or i >= self.cv_range[1]:
                yield t['id']


    #def get_learn_coordinate_routes(self):
        #for i in self._learn_range():
            #yield self._coordinate_routes[i]

    #def get_learn_routes(self):
        #for i in self._learn_range():
            #yield self._routes[i]

    #def get_validation_routes(self):
        #for i in self._validation_range():
            #yield self._routes[i]


    def get_learn_features(self):
        for i in self._learn_range():
            yield self.F.all_except('route', self._X[i])

    def get_validation_features(self):
        for i in self._validation_range():
            yield self.F.all_except('route', self._X[i])

    def get_features(self, a, features = set()):
        if not features or len(features) == 0:
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

    def road_id_to_index(self, road_id):
        return self.id_to_idx.get(road_id, None)

    def array_to_route(self, r):
        """
        Turn array into a set of road ids (that is without order)

        >>> r = list(range(100))
        >>> r2 = array_to_route( route_to_array(r) )
        >>> set(r) == set(r)
        True
        """
        return self.F.route.keys[self.F.route(r) != 0]

    #def array_to_point_pairs(self, r):
        #route = self.F.route(r)
        #point_pairs = [(tuple(sx), tuple(ex)) for sx, ex in self.idx_to_endpoints[route != 0, :]]
        #return point_pairs

    def route_to_point_pairs(self, r):
        return [self.id_to_point_pair[x] for x in r if x is not None]

    def distance(self, r1, r2, **kws):
        return self.F.distance(r1, r2, **kws)

    def nearest_neighbor(self, r, **kws):
        d_min = float('inf')
        r_min = None

        for r2 in self.get_learn_X():
            d = self.distance(r, r2, weights='uniform', **kws)
            if d < d_min:
                d_min = d
                r_min = r2

        return r_min, d_min


def extract_arc_coordinates(track):
    """
    return: { arcid => (start, end) }
    """

    r = {}
    arcid = None
    pos = None
    pos_new = None

    for arc in track['track']:
        arcid_new = arc['roadid']

        if arcid_new is not None:
            pos_new = (float(arc['lat']), float(arc['lon']))

        if arcid_new != arcid:
            # Leaving old arc
            if arcid is not None:
                r[arcid][1] = pos

            # Entering new arc
            arcid = arcid_new
            if arcid is not None:
                r[arcid] = [pos_new, pos_new]

        if pos_new is not None:
            pos = pos_new


    if arcid is not None:
        r[arcid][1] = pos

    return r

def merge_arc_coordinates(r1, r2):
    r = dict(r1)
    r.update(r2)
    return r


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

    return: matrix with shape (len(routes), len(ids)),
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



