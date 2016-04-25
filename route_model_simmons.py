

from collections import defaultdict, Counter

"""
Route model loosely modeled after the HMM approach from Simmons 2006 in
"Learning to Predict Driver Route and Destination Intent"
"""

class CyclicRouteException(Exception):
    pass


class RouteModelSimmons:

    ARRIVAL = None

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


    def _index(self, partial):
        return partial[-1]

    def learn_routes(self, routes, _):
        for route in routes:
            self._learn_route(route, route[-1])

    def _learn_route(self, route, g):
        """
        route: list of arc IDs
        """
        for i, (from_, to) in enumerate(zip(route, route[1:] + [self.ARRIVAL])):
            idx = self._index(route[:i+1])
            self._pgl[idx][g] += 1

            list_ = self._pls[idx]
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

        r = Counter()
        for l, g, m in self._pls[ partial_route[-1] ]:
            if fix_g is not None and g != fix_g:
                continue
            r[l] = arrivals[g] * m

        return r + Counter()

    def predict_route(self, partial_route):
        partial = partial_route[:]
        likelihood = 1.0
        arrivals = self.predict_arrival(partial_route)

        if len(arrivals) > 0:
            max_arrival = arrivals.most_common()[0][0]
        else:
            max_arrival = None

        arcs = {}
        while True:
            # MLE estimate, marginalize over goals
            most_likely = self.predict_arc(partial, fix_g = max_arrival).most_common()
            if len(most_likely) < 1:
                print("i'm lost!")
                break

            for i, (route_id, weight) in enumerate(most_likely):
                if route_id is self.ARRIVAL:
                    likelihood *= float(weight) / sum(v for _, v in most_likely)
                    return partial[len(partial_route):], likelihood

                elif route_id not in partial:
                    likelihood *= float(weight) / sum(v for _, v in most_likely)
                    partial.append(route_id)
                    break
            else:
                e = CyclicRouteException("no solution w/o cycle found, aborting route!")
                e.route = partial[len(partial_route):]
                raise e

        return partial[len(partial_route):], likelihood

