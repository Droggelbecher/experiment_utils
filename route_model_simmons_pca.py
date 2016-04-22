
from collections import defaultdict, Counter
import sys
import pprint

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN

import geo
import gmaps

class CyclicRouteException(Exception):
    pass

class RouteModelSimmonsPCA:

    ARRIVAL = None

    MAX_COMPONENTS = 3

    def __init__(self):
        # { (arc, pc0, pc1, ...): [(li, g, m), ...], ... }
        self._pls = defaultdict(list)

        # { (arc, pc0, pc1, ...): { g: m, ... }, ... }
        self._pgl = defaultdict(Counter)

    def _route_to_array(self, route, default = 0.0):
        s = set(route)
        r = np.full(len(self._road_id_to_index), default)
        for id_ in s:
            if id_ is None:
                # TODO: find out why this can even happen
                continue
            idx = self._road_id_to_index[id_]
            r[idx] = 1
        return r

    def _index(self, partial):
        """
        Turn given partial route into a hash table index
        """
        N_COMPONENTS = 1
        a = self._route_to_array(partial, default = 0.5)
        p = partial[-1] #if len(partial) else 0

        t = (p,) + tuple(self._quantize_pc(x) for x in self._pca.transform(a.reshape(1, -1))[0,:N_COMPONENTS])
        #t = p
        return t

    def _project(self, partial):
        a = self._route_to_array(partial, default = 0.5)
        return tuple(x for x in self._pca.transform(a.reshape(1, -1))[0])

    def _quantize_pc(self, v):
        #return v
        #return round(v, 1)
        eps = 0.1
        if v < -eps:
            return -1.0
        if v > eps:
            return 1.0
        return 0

    def _destination_label(self, road_id):
        return self._destination_labels(self._road_id_to_index(road_id))

    def learn_routes(self, routes, road_ids_to_endpoints):

        # Map: road_id (+dir) -> leave-point

        self._road_id_to_endpoint = {(k, 0): v[1] for k, v in road_ids_to_endpoints.items()}
        self._road_id_to_endpoint.update({(k, 1): v[0] for k, v in road_ids_to_endpoints.items()})

        # Map: road_id (+dir) -> index

        s = list(enumerate(sorted(road_ids_to_endpoints.keys())))
        l = len(s)
        self._road_id_to_index = {(k, 0): i for i, k in s}
        self._road_id_to_index.update({(k, 1): i + l for i, k in s})

        # Cluster destinations

        a_destinations = np.zeros((len(routes), 2))
        for i, route in enumerate(routes):
            a_destinations[i, :] = self._road_id_to_endpoint[route[-1]]

        dbscan = DBSCAN(eps = 200, metric = geo.distance).fit(a_destinations)

        print("destination labels", dbscan.labels_)

        g = gmaps.generate_gmaps(
                markers = a_destinations[dbscan.labels_ != -1, :],
                center = a_destinations[0, :],
                )
        f = open('/tmp/gmaps_destinations.html', 'w')
        f.write(g)
        f.close()


        # Principal component analysis -> correlated road ids

        # Convert routes to array
        # routes: [ [ (road_id, direction_flag), .... ], .... ]

        self._X = np.zeros(shape = (len(routes), len(self._road_id_to_index)))
        #self._X = np.full((len(routes), len(self._road_id_to_index)), -1.0)

        for i, route in enumerate(routes):
            for r in route:
                j = self._road_id_to_index[r]
                self._X[i, j] = 1

        self._pca = PCA(n_components = self.MAX_COMPONENTS)
        self._pca.fit(self._X)

        print("variances={}".format(self._pca.explained_variance_ratio_))

        print("learning routes...")
        for route, g in zip(routes, dbscan.labels_):
            self._learn_route(route, g)

        print "pgl="
        pprint.pprint(dict(self._pgl))
        print "pls="
        pprint.pprint(dict(self._pls))

    def _learn_route(self, route, g):
        #g = self._destination_label(route[-1])

        if g == -1:
            print("route classified as noise")
            # destination was classified as noise, don't learn this route!
            return

        for i, (from_, to) in enumerate(zip(route, route[1:] + [self.ARRIVAL])):
            pc_vector = self._index(route[:i + 1])

            #print("g=", g)
            self._pgl[ pc_vector ][g] += 1
            #print("-----> pgl=", self._pgl)
            list_ = self._pls[pc_vector]

            for i, (to2, g2, m) in enumerate(list_):
                if to2 == to and g2 == g:
                    list_[i] = (to, g, m + 1)
                    break
            else:
                list_.append( (to, g, 1) )

    def predict_arrival(self, partial_route):
        return self._pgl[ self._index(partial_route) ]

    def predict_arc(self, partial_route, fix_g = None):
        """
        returns: { route_id: count, ... }
        """
        arrivals = self.predict_arrival(partial_route)

        pc_weights = self._pca.inverse_transform(self._project(partial_route))

        r = Counter()
        for l, g, m in self._pls[ self._index(partial_route) ]:

            if fix_g is not None and g != fix_g:
                continue

            if l is self.ARRIVAL:
                w = 1.0
            else:
                w = pc_weights[self._road_id_to_index[l]]

            if w < 0:
                w = 0
            #print('r[{}] = {:6.4f} * {:6.4f} * {:6.4f}'.format(l, arrivals[g], m, w))

            r[l] = arrivals[g] * m * w

        return r


    def predict_route(self, partial_route):
        partial = partial_route[:]

        confidence = 1.0


        arrivals = self.predict_arrival(partial_route)
        if len(arrivals) > 0:
            max_arrival = arrivals.most_common()[0][0]
        else:
            max_arrival = None

        arcs = {}
        while True:
            most_likely = self.predict_arc(partial, fix_g = max_arrival).most_common()
            if len(most_likely) < 1:
                # TODO Should being lost lower our confidence?
                print("i'm lost!")
                confidence = 0
                break

            for i, (route_id, weight) in enumerate(most_likely):
                if route_id is None:
                    #d = float(weight - (most_likely[1][1] if len(most_likely) > 1 else 0.0))
                    confidence *= (float(weight) / sum(v for _, v in most_likely))
                    return partial[len(partial_route):], confidence

                elif route_id not in partial:
                    partial.append(route_id)
                    break
            else:
                e = CyclicRouteException("no solution w/o cycle found, aborting route!")
                e.route = partial[len(partial_route):]
                raise e

            confidence *= float(weight) / sum(v for _, v in most_likely)
            #d = float(weight - (most_likely[1][1] if len(most_likely) > 1 else 0.0))
            #confidence *= weight

        r = partial[len(partial_route):], confidence
        return partial[len(partial_route):], confidence


