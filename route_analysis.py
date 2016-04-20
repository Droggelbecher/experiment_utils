#!/usr/bin/env python

import numpy as np
import scipy
import scipy.sparse
import csv
import sys
from datetime import datetime
import math
from collections import Counter

from features import Features, Feature
import geo

csv.field_size_limit(sys.maxsize)

class Routes:

    # TODO: Having a fixed max distance is probably not so good (as it gives a strong bias to the distance metric),
    # rather try scaling to the max distance in the dataset?

    # in m, used for distance between eg. 2 destinations
    MAX_CONCEIVABLE_DISTANCE = 10*1000

    def __init__(self, d):
        routes = d['routes']
        road_ids_to_endpoints = d['road_ids_to_endpoints']
        coordinate_routes = d['coordinate_routes']

        sorted_road_ids = np.array(sorted(road_ids_to_endpoints.keys()))

        self.F = Features(
                Feature('weekdays',  ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'], weight = 0.0),
                Feature('hours',     np.arange(0, 24, 0.5),                      weight = 0.0),
                Feature('arrival',   ('lat', 'lon'),                             weight = 0.2),
                Feature('departure', ('lat', 'lon'),                             weight = 0.0),
                Feature('route',     sorted_road_ids,                            weight = 0.8),
                Feature('arrival_arcs', sorted_road_ids,                         weight = 0.0),
                )

        self.endpoints = np.array([road_ids_to_endpoints[id_] for id_ in sorted_road_ids])

        a_arrival = np.array([c[-1] for c in coordinate_routes])
        a_departure = np.array([c[0] for c in coordinate_routes])

        # Departure Time

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

        #self.sorted_road_ids = sorted_road_ids
        a_road_ids, a_arrival_arcs = routes_to_array(routes, sorted_road_ids)

        # TODO: this needs to stay consistent with the offsets above
        self.X = np.hstack((
            a_weekdays,
            a_hours,
            a_arrival,
            a_departure,
            a_road_ids,
            a_arrival_arcs
            ))

    @staticmethod
    def _jaccard_dist(a, b):
        intersection = np.sum(a * b)
        union = np.sum(a + b)
        return 1.0 - float(intersection) / float(union)


    def _nonroute_dist(self, a, b):
        dist_hours = abs(self.F.hours.b_mean(a) - self.F.hours.b_mean(b)) / 24.0
        dist_weekdays = 0.0 if np.all(self.F.weekdays(a) == self.F.weekdays(b)) else 1.0
        dist_arrival = geo.distance(self.F.arrival(a), self.F.arrival(b)) / self.MAX_CONCEIVABLE_DISTANCE
        dist_arrival = min(dist_arrival, 1.0)
        dist_departure = geo.distance(self.F.departure(a), self.F.departure(b)) / self.MAX_CONCEIVABLE_DISTANCE
        dist_departure = min(dist_departure, 1.0)

        return (dist_hours * self.F.hours.weight
                + dist_weekdays * self.F.weekdays.weight
                + dist_arrival * self.F.arrival.weight
                + dist_departure * self.F.departure.weight)

    def distance_jaccard(self, r1, r2):
        if len(r1) == self.X.shape[1]:
            nr = self._nonroute_dist(r1, r2) 
            j = Routes._jaccard_dist(self.F.route(r1), self.F.route(r2)) * self.F.route.weight
            return nr + j
        else:
            return 0.0

def routes_to_array(routes, ids):
    """
    routes: iterable over (iterable over route ids)
    ids: array of all possible road ids, sorted

    return: sparse matrix with shape (len(routes), len(ids)),
        containing 1 if road id belongs to that route
    """
    #a = scipy.sparse.lil_matrix(0, (len(routes), len(ids)), dtype=np.bool_)
    a = np.zeros(shape=(len(routes), len(ids)))
    a_arrival = np.zeros(shape=(len(routes), len(ids)))

    for i, route in enumerate(routes):
        for r in route:
            # linear search to find route index from route id
            for j, id_ in enumerate(ids):
                if id_ == r:
                    a[i, j] = 1
                    break
        for j, id_ in enumerate(ids):
            if id_ == route[-1]:
                a_arrival[i, j] = 1
    return a, a_arrival




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



