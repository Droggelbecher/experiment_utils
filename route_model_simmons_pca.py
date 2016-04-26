
from collections import defaultdict, Counter
import sys
import pprint

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN

import geo
import gmaps

from route_model_simmons import RouteModelSimmons

class RouteModelSimmonsPCA(RouteModelSimmons):

    ARRIVAL = None

    #MAX_COMPONENTS = 3
    PCA_WEIGHTS = False
    CLUSTER_DESTINATIONS = False
    REJECT_NOISE_DESTINATIONS = False
    USE_ICA = False
    #INDEX_COMPONENTS = 2


    def __init__(self, pca_components = 3):
        self.MAX_COMPONENTS = pca_components
        self.INDEX_COMPONENTS = pca_components
        RouteModelSimmons.__init__(self)

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

        if self.INDEX_COMPONENTS == 0:
            return RouteModelSimmons._index(self, partial, features)

        else:

            a = np.hstack((self._route_to_array(partial, default = 0.5), np.array(features)))
            if len(partial):
                p = partial[-1]
            else:
                p = -1
            r = (p,) + tuple(self._quantize_pc(x) for x in self._pca.transform(a.reshape(1, -1))[0,:self.INDEX_COMPONENTS])
            return r

    def _project(self, partial, features):
        a = np.hstack((self._route_to_array(partial, default = 0.5), np.array(features)))
        #return tuple(x for x in self._pca.transform(a.reshape(1, -1))[0])
        return self._pca.transform(a.reshape(1, -1))

    def _quantize_pc(self, v):
        #return 0
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

        if self.CLUSTER_DESTINATIONS:
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

        parts = 4
        self._X = np.zeros(shape = (len(routes) * parts, len(self._road_id_to_index) + len(dummyfeature)))

        for i, routefeatures in enumerate(routes):
            route, features = self._split_route(routefeatures)
            if not len(route):
                continue

            n = int(len(route)/parts)
            for part in range(parts):
                if part < parts - 1:
                    route = route[:part * n]
                for r in route:
                    j = self._road_id_to_index[r]
                    self._X[i, j] = 1

                self._X[i, len(self._road_id_to_index):] = features

        if self.USE_ICA:
            self._pca = FastICA(n_components = self.MAX_COMPONENTS)
        else:
            self._pca = PCA(n_components = self.MAX_COMPONENTS)

        self._pca.fit(self._X)

        if hasattr(self._pca, 'explained_variance_ratio_'):
            print("variances={}".format(self._pca.explained_variance_ratio_))

        print("learning routes...")
        if self.CLUSTER_DESTINATIONS:
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
        pc_weights = self._pca.inverse_transform(self._project(partial_route, features))
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




