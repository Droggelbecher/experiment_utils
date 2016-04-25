
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

    MAX_COMPONENTS = 3
    PCA_WEIGHTS = False
    CLUSTER_DESTINATIONS = False
    REJECT_NOISE_DESTINATIONS = False
    INDEX_COMPONENTS = 0

    def _route_to_array(self, route, default = 0.0):
        s = set(route)
        r = np.full(len(self._road_id_to_index), default)
        for id_ in s:
            #if id_ is None:
                ## TODO: find out why this can even happen
                #continue
            idx = self._road_id_to_index[id_]
            r[idx] = 1
        return r

    def _index(self, partial):
        """
        Turn given partial route into a hash table index
        """
        a = self._route_to_array(partial, default = 0.5)
        p = partial[-1] #if len(partial) else 0

        t = (p,) + tuple(self._quantize_pc(x) for x in self._pca.transform(a.reshape(1, -1))[0,:self.INDEX_COMPONENTS])
        #t = p
        return t

    def _project(self, partial):
        a = self._route_to_array(partial, default = 0.5)
        return tuple(x for x in self._pca.transform(a.reshape(1, -1))[0])

    def _quantize_pc(self, v):
        eps = 0.1
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

        for i, route in enumerate(routes):
            for r in route:
                j = self._road_id_to_index[r]
                self._X[i, j] = 1

        self._pca = PCA(n_components = self.MAX_COMPONENTS)
        self._pca.fit(self._X)

        print("variances={}".format(self._pca.explained_variance_ratio_))

        print("learning routes...")
        if self.CLUSTER_DESTINATIONS:
            for route, g in zip(routes, dbscan.labels_):
                self._learn_route(route, g)
        else:
            RouteModelSimmons.learn_routes(self, routes, road_ids_to_endpoints)

    def _learn_route(self, route, g):
        if g == -1 and self.REJECT_NOISE_DESTINATIONS:
            print("route classified as noise")
            return

        RouteModelSimmons._learn_route(self, route, g)

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

            if not self.PCA_WEIGHTS or l is self.ARRIVAL:
                w = 1.0
            else:
                w = pc_weights[self._road_id_to_index[l]]

            w = max(0, w)
            #print('r[{}] = {:6.4f} * {:6.4f} * {:6.4f}'.format(l, arrivals[g], m, w))

            r[l] = arrivals[g] * m * w

        # The "... + Counter()" is for normalization (remove 0-counts etc...)
        return r + Counter()




