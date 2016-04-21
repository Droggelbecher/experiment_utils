

from collections import defaultdict, Counter

"""
Route model loosely modeled after the HMM approach from Simmons 2006 in
"Learning to Predict Driver Route and Destination Intent"
"""


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

    def learn_route(self, route):
        """
        route: list of arc IDs
        """
        g = route[-1]

        for from_, to in zip(route, route[1:] + [self.ARRIVAL]):
            self._pgl[from_][g] += 1

            list_ = self._pls[from_]
            for i, (to2, g2, m) in enumerate(list_):
                if to2 == to and g2 == g:
                    list_[i] = (to, g, m + 1)
                    break
            else:
                list_.append( (to, g, 1) )

    def predict_arrival(self, partial_route):
        return self._pgl[ partial_route[-1] ]

    def predict_arc(self, partial_route):
        """
        returns: { route_id: count, ... }
        """
        arrivals = self.predict_arrival(partial_route)

        r = Counter()
        for l, g, m in self._pls[ partial_route[-1] ]:
            r[l] = arrivals[g] * m

        return r

    def predict_route(self, partial_route):
        partial = partial_route[:]

        arcs = {}
        while True:
            # MLE estimate, marginalize over goals
            most_likely = self.predict_arc(partial).most_common(1)
            if len(most_likely) < 1 or most_likely[0][0] is None:
                break

            partial.append(most_likely[0][0])

            if len(partial) > 1000:
                print("too long")
                break

        return partial[len(partial_route):]

