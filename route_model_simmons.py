
import numpy as np

from collections import defaultdict, Counter

"""
Route model loosely modeled after the HMM approach from Simmons 2006 in
"Learning to Predict Driver Route and Destination Intent"
"""

class RouteException(Exception):
    pass


class RouteModelSimmons:

    ARRIVAL = None
    BACKTRACK = True

    def __init__(self):
        # p(li|sj) = { lj: [(li, g, m), ...], ... }
        # li = "to" road id
        # lj = "from" road id
        # g = goal = arrival
        # m = count
        self._pls = defaultdict(list)

        # p(g|l) = { l: { g: m, ... }, ... }
        # g = goal = arrival
        # l = road id
        # m = count
        self._pgl = defaultdict(Counter)

        self._accept_wrong_arrival = True

    def _index(self, partial, features):
        if len(partial):
            return (partial[-1],) + tuple(features)
        else:
            return (-1,) + tuple(features)

    def _route_to_array(self, route, default = 0.0):
        s = set(route)
        r = np.full(len(self._road_id_to_index), default)
        for id_ in s:
            idx = self._road_id_to_index[id_]
            r[idx] = 1
        return r

    def learn_routes(self, routes, features, road_ids_to_endpoints):

        # Map: road_id (+dir) -> index

        s = list(enumerate(sorted(road_ids_to_endpoints.keys())))
        l = len(s)
        self._road_id_to_index = {(k, 0): i for i, k in s}
        self._road_id_to_index.update({(k, 1): i + l for i, k in s})

        # TODO: for this model its overkill to store X completely,
        # we can compute the average continuously

        X = np.zeros(shape = (len(routes), len(self._road_id_to_index) + features.shape[1]))
        for i, route in enumerate(routes):
            if not len(route):
                continue
            X[i,features.shape[1]:] = self._route_to_array(route)
        X[:,:features.shape[1]] = features
        self._average = np.average(X, axis=0)

        for i, route in enumerate(routes):
            if not len(route):
                continue
            self._learn_route(route, route[-1], features[i,:])

    def _learn_route(self, route, g, features):
        """
        route: list of arc IDs
        """
        if not len(route):
            return
        for i, (from_, to) in enumerate(zip(route, route[1:] + [self.ARRIVAL])):
            idx = self._index(route[:i+1], features)
            self._pgl[idx][g] += 1

            list_ = self._pls[idx]
            for i, (to2, g2, m) in enumerate(list_):
                if to2 == to and g2 == g:
                    list_[i] = (to, g, m + 1)
                    break
            else:
                list_.append( (to, g, 1) )

    def predict_arrival(self, partial_route, features):
        return self._pgl[ self._index(partial_route, features) ]

    def predict_arc(self, partial_route, features, fix_g = None):
        """
        returns: { route_id: count, ... }
        """
        arrivals = self.predict_arrival(partial_route, features)

        r = Counter()
        for l, g, m in self._pls[ self._index(partial_route, features) ]:
            if fix_g is not None and g != fix_g:
                continue
            r[l] = arrivals[g] * m

        return r + Counter()

    def _predict_route(self, partial_route, features):
        partial = partial_route[:]
        likelihood = 1.0
        arrivals = self.predict_arrival(partial_route, features)

        if len(arrivals) > 0:
            max_arrival = arrivals.most_common()[0][0]
        else:
            max_arrival = None

        forbidden_arcs = set()
        while True:
            #print(partial[len(partial_route):])
            # MLE estimate, marginalize over goals
            most_likely = self.predict_arc(partial, features, fix_g = max_arrival).most_common()
            likely_allowed = [(k, v) for (k, v) in most_likely if k not in forbidden_arcs]

            for i, (route_id, weight) in enumerate(likely_allowed):
                if route_id is self.ARRIVAL:
                    if max_arrival is None or route_id == max_arrival or self._accept_wrong_arrival:
                        print("arrival found. correct={}".format(route_id == max_arrival))
                        likelihood *= float(weight) / sum(v for _, v in most_likely)
                        return partial[len(partial_route):], likelihood
                    else:
                        #print("wrong arrival found")
                        continue

                elif route_id not in partial:
                    #print("normal arc")
                    likelihood *= float(weight) / sum(v for _, v in most_likely)
                    partial.append(route_id)
                    break
            else:
                if not self.BACKTRACK or len(partial) <= len(partial_route):
                    print("[E] stuck with no alternative")
                    print("partial[-1]:", partial[-1])
                    print("partial[l:]:", partial[len(partial_route):])
                    print("max. arrival:", max_arrival)
                    print("allowed:", likely_allowed)
                    print("forbidden:", forbidden_arcs)
                    e = RouteException("stuck with no alternatives!")
                    e.route = partial[len(partial_route):]
                    raise e
                else:
                    #print("backtracking")
                    forbidden_arcs.add(partial[-1])
                    del partial[-1]
                    continue

        return partial[len(partial_route):], likelihood

    def predict_route(self, partial_route, features):
        # TODO: Quick hack that throws away likelihood and substitutes it with
        # average-projection
        route, likeli = self._predict_route(partial_route, features)

        queryvector = np.hstack((features, self._route_to_array(route, default = 0.0)))
        dot = queryvector.dot(self._average)
        #norm = np.linalg.norm(queryvector) * np.linalg.norm(self._average)
        norm = (len(route) + 1.0) * np.linalg.norm(self._average)

        return (route, dot / norm)

