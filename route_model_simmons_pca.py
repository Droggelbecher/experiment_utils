
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
    REJECT_NOISE_ARRIVALS = False

    def __init__(self, decompositor = PCA(n_components = 3),
            cluster_arrivals = False,
            cluster_departures = False):
        self._decompositor = decompositor
        self._pca_route_parts = 1
        self._cluster_arrivals = cluster_arrivals
        self._cluster_departures = cluster_departures
        RouteModelSimmons.__init__(self)
        self._accept_wrong_arrival = not self._cluster_arrivals

    def _route_to_array(self, route, default = 0.0):
        s = set(route)
        r = np.full(len(self._road_id_to_index), default)
        for id_ in s:
            idx = self._road_id_to_index[id_]
            r[idx] = 1
        return r

    def _index(self, partial, features):
        """
        Turn given partial route into a hash table index
        """

        if self._decompositor is None:
            return RouteModelSimmons._index(self, partial, features)

        else:
            a = np.hstack((features, self._route_to_array(partial, default = 0.0)))
            if len(partial):
                p = partial[-1]
            else:
                p = -1
            
            transformed = self._decompositor.transform(a.reshape(1, -1))[0, :]
            r = (p,) + tuple(self._quantize_pc(x) for x in transformed)
            return r

    def _project(self, partial, features):
        a = np.hstack((features, self._route_to_array(partial, default = 0.0)))
        return self._decompositor.transform(a.reshape(1, -1))

    def _quantize_pc(self, v):
        eps = .1
        if v < -eps:
            return -1.0
        if v > eps:
            return 1.0
        return 0

    def learn_routes(self, routes, features_, road_ids_to_endpoints):

        # Map: road_id (+dir) -> leave-point

        self._road_id_to_endpoint = {(k, 0): v[1] for k, v in road_ids_to_endpoints.items()}
        self._road_id_to_endpoint.update({(k, 1): v[0] for k, v in road_ids_to_endpoints.items()})

        # Map: road_id (+dir) -> index

        s = list(enumerate(sorted(road_ids_to_endpoints.keys())))
        l = len(s)
        self._road_id_to_index = {(k, 0): i for i, k in s}
        self._road_id_to_index.update({(k, 1): i + l for i, k in s})

        # Cluster departures

        features = features_
        if self._cluster_departures:
            #a_departures = np.zeros((len(routes), 2))
            for i, route in enumerate(routes):
                if not len(route):
                    continue
                a_departures[i, :] = self._road_id_to_endpoint[route[0]]

            dbscan = DBSCAN(eps = 200, metric = geo.distance).fit(a_departures)

            print("departure labels", dbscan.labels_)

            g = gmaps.generate_gmaps(
                    markers = a_departures[dbscan.labels_ != -1, :],
                    center = a_departures[0, :],
                    )
            f = open('/tmp/gmaps_departures.html', 'w')
            f.write(g)
            f.close()

            features = np.zeros((len(routes), features_.shape[1] + 1))
            features[:, :-1] = features_
            features[:, -1:] = dbscan.labels_


        # Cluster arrivals

        if self._cluster_arrivals:
            a_arrivals = np.zeros((len(routes), 2))
            for i, route in enumerate(routes):
                if not len(route):
                    continue
                a_arrivals[i, :] = self._road_id_to_endpoint[route[-1]]

            dbscan = DBSCAN(eps = 200, metric = geo.distance).fit(a_arrivals)

            print("arrival labels", dbscan.labels_)

            g = gmaps.generate_gmaps(
                    markers = a_arrivals[dbscan.labels_ != -1, :],
                    center = a_arrivals[0, :],
                    )
            f = open('/tmp/gmaps_arrivals.html', 'w')
            f.write(g)
            f.close()


        # Principal component analysis -> correlated road ids

        # Convert routes to array
        # routes: [ [ (road_id, direction_flag), .... ], .... ]


        parts = self._pca_route_parts
        self._X = np.zeros(shape = (len(routes) * parts, len(self._road_id_to_index) + features.shape[1]))

        rta = self._route_to_array
        X = self._X

        for i, route in enumerate(routes):
            if not len(route):
                continue

            if parts == 1:
                X[i,features.shape[1]:] = rta(route)

            else:
                n = int(len(route)/parts)
                for part in range(parts):
                    if part < parts - 1:
                        route2 = route[:(part+1) * n]

                    X[i * parts + part, features.shape[1]:] = rta(route2)

        X[:,:features.shape[1]] = features
        
        self._average = np.average(X, axis=0)

        self._decompositor.fit(self._X)

        if hasattr(self._decompositor, 'explained_variance_ratio_'):
            print("variances={}".format(self._decompositor.explained_variance_ratio_))

        print("learning routes...")
        if self._cluster_arrivals:
            for i, (route, g) in enumerate(zip(routes, dbscan.labels_)):
                self._learn_route(route, g, features[i,:])
        else:
            RouteModelSimmons.learn_routes(self, routes, features, road_ids_to_endpoints)

    def _learn_route(self, route, g, features):
        if g == -1 and self.REJECT_NOISE_ARRIVALS:
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
            r[l] = arrivals[g] * m * w

        # The "... + Counter()" is for normalization (remove 0-counts etc...)
        return r + Counter()

    def predict_route(self, partial_route, features):
        route, likeli = RouteModelSimmons.predict_route(self, partial_route, features)

        queryvector = np.hstack((features, self._route_to_array(route, default = 0.0)))
        dot = queryvector.dot(self._average)
        norm = (np.linalg.norm(queryvector) * np.linalg.norm(self._average))

        return (route, dot / norm)




