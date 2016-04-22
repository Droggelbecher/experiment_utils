
from collections import defaultdict, Counter
import numpy as np
from sklearn.decomposition import PCA, FastICA
import sys

class CyclicRouteException(Exception):
    pass

class RouteModelSimmonsPCA:

    ARRIVAL = None

    def __init__(self):
        # { (arc, pc0, pc1, ...): [(li, g, m), ...], ... }
        self._pls = defaultdict(list)

        # { (arc, pc0, pc1, ...): { g: m, ... }, ... }
        self._pgl = defaultdict(Counter)

    def _route_to_array(self, route, default = 0.0):
        s = set(route)
        r = np.full(len(self._sorted_road_ids), default)
        for id_ in s:
            if id_ is None:
                # TODO: find out why this can even happen
                continue
            idx = self._sorted_road_ids.index(id_) 
            r[self._sorted_road_ids.index(id_)] = 1
        return r

    def _project(self, partial):
        """
        Project given (partial) route to principal components
        """
        a = self._route_to_array(partial, default = 0.1)
        p = partial[-1] if len(partial) else 0
        t = (p,) + tuple(round(x, 1) for x in self._pca.transform(a.reshape(1, -1))[0])
        #print("t=", t)
        return t


    def learn_routes(self, routes, road_ids_to_endpoints):
        print("pca...")
        #routes = d['routes']
        #road_ids_to_endpoints = d['road_ids_to_endpoints']
        #coordinate_routes = d['coordinate_routes']
        self._sorted_road_ids = [(x, 0) for x in sorted(road_ids_to_endpoints.keys())] \
                + [(x, 1) for x in sorted(road_ids_to_endpoints.keys())]


        # Convert routes to array
        # routes: [ [ (road_id, direction_flag), .... ], .... ]

        self._X = np.zeros(shape = (len(routes), len(self._sorted_road_ids)))

        for i, route in enumerate(routes):
            for r in route:
                try:
                    j = self._sorted_road_ids.index(r)
                    self._X[i, j] = 1
                except ValueError:
                    pass


        MAX_COMPONENTS = 3

        self._pca = PCA(n_components = MAX_COMPONENTS)
        self._pca.fit(self._X)

        print("learning routes...")
        for i, route in enumerate(routes):
            #print('learn {}/{}'.format(i, len(routes)))
            #sys.stdout.flush()
            self._learn_route(route)

    def _learn_route(self, route):
        g = route[-1]

        for i, (from_, to) in enumerate(zip(route, route[1:] + [self.ARRIVAL])):
            pc_vector = self._project(route[:i])

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
        #print "pgl=", self._pgl
        return self._pgl[ self._project(partial_route) ]

    def predict_arc(self, partial_route):
        """
        returns: { route_id: count, ... }
        """
        arrivals = self.predict_arrival(partial_route)
        #print("arrivals=", arrivals)
        r = Counter()
        for l, g, m in self._pls[ self._project(partial_route) ]:
            r[l] = arrivals[g] * m

        return r


    def predict_route(self, partial_route):
        partial = partial_route[:]

        arcs = {}
        while True:
            # MLE estimate, marginalize over goals
            most_likely = self.predict_arc(partial).most_common()
            if len(most_likely) < 1:
                print("i'm lost!")
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


