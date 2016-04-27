
from collections import defaultdict, Counter
import sys
import pprint

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN

import geo
import gmaps
import plots

from route_model_simmons import RouteModelSimmons

class RouteModelSimmonsPCA(RouteModelSimmons):

    ARRIVAL = None

    PCA_WEIGHTS = False
    REJECT_NOISE_DESTINATIONS = False

    def __init__(self, decompositor = PCA(n_components = 3),
            cluster_destinations = False):
        self._decompositor = decompositor
        self._pca_route_parts = 1
        self._cluster_destinations = cluster_destinations
        RouteModelSimmons.__init__(self)
        self._accept_wrong_arrival = not self._cluster_destinations

    def _route_to_array(self, route, features, default = 0.0):
        s = set(route)
        r = np.full(len(self._road_id_to_index), default)
        for id_ in s:
            idx = self._road_id_to_index[id_]
            r[idx] = 1
        return np.hstack((np.array(features), r))
        #return np.hstack((np.zeros(len(self._road_id_to_index)), np.array(features)))

    def _index(self, partial, features):
        """
        Turn given partial route into a hash table index
        """

        if self._decompositor is None:
            return RouteModelSimmons._index(self, partial, features)

        else:

            a = self._route_to_array(partial, default = 0.0, features = features)
            if len(partial):
                p = partial[-1]
            else:
                p = -1
            
            #print("a=", a)
            transformed = self._decompositor.transform(a.reshape(1, -1))[0, :]
            #print("transformed=", transformed)

            r = (p,) + tuple(self._quantize_pc(x) for x in transformed)
            return r

    def _project(self, partial, features):
        a = self._route_to_array(partial, default = 0.5, features = features)
        return self._decompositor.transform(a.reshape(1, -1))

    def _quantize_pc(self, v):
        #return round(v, 1)
        eps = .1
        if v < -eps:
            return -1.0
        if v > eps:
            return 1.0
        return 0

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

        if self._cluster_destinations:
            a_destinations = np.zeros((len(routes), 2))
            for i, routefeatures in enumerate(routes):
                route, features = self._split_route(routefeatures)
                if not len(route):
                    continue
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


        dummyroute, dummyfeature = self._split_route(routes[0])

        parts = self._pca_route_parts
        self._X = np.zeros(shape = (len(routes) * parts, len(self._road_id_to_index) + len(dummyfeature)))

        rta = self._route_to_array
        X = self._X

        for i, routefeatures in enumerate(routes):
            route, features = self._split_route(routefeatures)
            #print("len(route)=", len(route), " parts=", parts)
            if not len(route):
                continue

            if parts == 1:
                X[i,:] = rta(route, features)

            else:
                n = int(len(route)/parts)
                for part in range(parts):
                    #print('route=', len(route))
                    if part < parts - 1:
                        route2 = route[:(part+1) * n]

                    #print('part=', part, 'n=', n, 'route=', len(route2))

                    X[i * parts + part, :] = rta(route2, features)

        self._decompositor.fit(self._X)

        if hasattr(self._decompositor, 'explained_variance_ratio_'):
            print("variances={}".format(self._decompositor.explained_variance_ratio_))

        print("learning routes...")
        if self._cluster_destinations:
            for routefeatures, g in zip(routes, dbscan.labels_):
                route, features = self._split_route(routefeatures)
                self._learn_route(route, g, features)
        else:
            RouteModelSimmons.learn_routes(self, routes, road_ids_to_endpoints)

    def _learn_route(self, route, g, features):
        if g == -1 and self.REJECT_NOISE_DESTINATIONS:
            print("route classified as noise")
            return

        RouteModelSimmons._learn_route(self, route, g, features)

    def predict_arc(self, partial_route, features, fix_g = None):
        """
        returns: { route_id: count, ... }
        """
        arrivals = self.predict_arrival(partial_route, features)
        if self.PCA_WEIGHTS:
            pc_weights = self._decompositor.inverse_transform(self._project(partial_route, features))
        r = Counter()

        for l, g, m in self._pls[ self._index(partial_route, features) ]:
            if fix_g is not None and g != fix_g:
                continue

            if not self.PCA_WEIGHTS or l is self.ARRIVAL:
                w = 1.0
            else:
                w = pc_weights[self._road_id_to_index[l]]

            w = max(0, w)
            #print('r[{}] = {:6.4f} * {:6.4f} * {:6.4f}'.format(l, arrivals[g], m, w))

            r[l] = arrivals[g] * m * w

        # The "... + Counter()" is for normalization (remove 0-counts etc...)
        return r + Counter()




