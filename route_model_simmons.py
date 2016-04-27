

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

    def _split_route(self, route):
        return route[0], route[1]

    def _index(self, partial, features):
        if len(partial):
            return (partial[-1],) + tuple(features)
        else:
            return (-1,) + tuple(features)

    def learn_routes(self, routes, _):
        for r in routes:
            route, features = self._split_route(r)
            if not len(route):
                continue
            self._learn_route(route, route[-1], features)

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

    def predict_route(self, partial_route, features):
        partial = partial_route[:]
        likelihood = 1.0
        arrivals = self.predict_arrival(partial_route, features)

        #print("-- predict_route")
        #print("index=", self._index(partial_route, features))
        #print("partial=", partial_route)
        #print("likely arrivals=", arrivals)

        if len(arrivals) > 0:
            max_arrival = arrivals.most_common()[0][0]
        else:
            max_arrival = None

        #print("max_arrival=", max_arrival)

        forbidden_arcs = set()

        #print("max_arrival", max_arrival)

        while True:
            #print(partial[len(partial_route):])
            # MLE estimate, marginalize over goals
            most_likely = self.predict_arc(partial, features, fix_g = max_arrival).most_common()
            likely_allowed = [(k, v) for (k, v) in most_likely if k not in forbidden_arcs]

            #print("most_likely", most_likely)
            #print("likely_allowed", likely_allowed)


            if len(likely_allowed) < 1:
                if not self.BACKTRACK or len(partial) <= len(partial_route):
                    #print("stuck with no alternative")
                    #return partial[len(partial_route):], likelihood
                    e = RouteException("stuck with no alternatives!")
                    e.route = partial[len(partial_route):]
                    raise e
                else:
                    #print("backtracking")
                    forbidden_arcs.add(partial[-1])
                    del partial[-1]
                    continue

            for i, (route_id, weight) in enumerate(likely_allowed):
                if route_id is self.ARRIVAL:
                    if max_arrival is None or route_id == max_arrival:
                        #print("max arrival found")
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
                print("no cycle-free solution")
                likelihood *= float(weight) / sum(v for _, v in most_likely)
                #return partial[len(partial_route):], likelihood
                e = RouteException("no solution w/o cycle found, aborting route!")
                e.route = partial[len(partial_route):]
                raise e

        return partial[len(partial_route):], likelihood

