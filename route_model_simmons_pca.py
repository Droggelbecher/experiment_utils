
from collections import defaultdict, Counter
import numpy as np
from sklearn.decomposition import PCA, FastICA
import sys

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
        a = self._route_to_array(partial, default = 0.0)
        p = partial[-1] #if len(partial) else 0
        #t = (p,) + tuple(self._quantize_pc(x) for x in self._pca.transform(a.reshape(1, -1))[0])
        #print("t=", t)

        t = (p,) + tuple(self._quantize_pc(x) for x in self._pca.transform(a.reshape(1, -1))[0,:1])
        #t = p
        return t

    def _project(self, partial):
        a = self._route_to_array(partial, default = 0.0)
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


    def learn_routes(self, routes, road_ids_to_endpoints):
        print("pca...")
        #routes = d['routes']
        #road_ids_to_endpoints = d['road_ids_to_endpoints']
        #coordinate_routes = d['coordinate_routes']

        s = list(enumerate(sorted(road_ids_to_endpoints.keys())))
        l = len(s)

        self._road_id_to_index = {(k, 0): i for i, k in s}
        self._road_id_to_index.update({(k, 1): i + l for i, k in s})


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
        for i, route in enumerate(routes):
            #print('learn {}/{}'.format(i, len(routes)))
            #sys.stdout.flush()
            self._learn_route(route)

    def _learn_route(self, route):
        g = route[-1]

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

    def predict_arc(self, partial_route):
        """
        returns: { route_id: count, ... }
        """
        arrivals = self.predict_arrival(partial_route)

        pc_weights = self._pca.inverse_transform(self._project(partial_route))

        r = Counter()
        for l, g, m in self._pls[ self._index(partial_route) ]:
            if l is self.ARRIVAL:
                w = 1.0
            else:
                w = pc_weights[self._road_id_to_index[l]]
            #print('r[{}] = {:6.4f} * {:6.4f} * {:6.4f}'.format(l, arrivals[g], m, w))

            r[l] = arrivals[g] * m * w

        return r


    def predict_route(self, partial_route):
        partial = partial_route[:]

        arcs = {}
        while True:
            # MLE estimate, marginalize over goals
            most_likely = self.predict_arc(partial).most_common()
            if len(most_likely) < 1:
                print("i'm lost!")
                #print("lost after: ", partial[len(partial_route):])
                break

            for i, m in enumerate(most_likely):
                if m[0] is None or m[0][0] is None:
                    print("is None: [{}]={}".format(i, m))
                    return partial[len(partial_route):]

                elif m[0] not in partial:
                    partial.append(m[0])
                    break
            else:
                e = CyclicRouteException("no solution w/o cycle found, aborting route!")
                e.route = partial[len(partial_route):]
                raise e

        return partial[len(partial_route):]


