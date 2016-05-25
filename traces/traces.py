
import random

class Trace:
    def __init__(self, **kws):
        self.__dict__.update(kws)

class Traces:

    def __init__(self, tracedata = None):

        self._traces = []
        self._road_ids_to_endpoints = {}

        if tracedata is not None:
            d = zip(
                    tracedata['routes'],
                    tracedata['coordinate_routes'],
                    tracedata['departure_times']
                    )

            for route, coordinate_route, departure_time in d:
                trace = Trace(
                        route = route,
                        coordinate_route = coordinate_route,
                        departure_time = departure_time
                        )
                self._traces.append(trace)

            self._road_ids_to_endpoints = tracedata['road_ids_to_endpoints']

        self._direct_road_ids_to_endpoints()

    def __iter__(self):
        return iter(self._traces)

    def __delitem__(self, i):
        del self._traces[i]

    def _direct_road_ids_to_endpoints(self):
        if type(self._road_ids_to_endpoints.keys()[0]) is tuple \
                and type(self._road_ids_to_endpoints.values()[0]) is tuple:
            return

        self._road_ids_to_endpoints = {
                (id_, dir_): (ep[dir_], ep[1 - dir_])
                for dir_ in (0, 1)
                for id_, ep in self._road_ids_to_endpoints.items()
                }

    def shuffle(self, seed = None):
        random.shuffle(self._traces, seed)

    def assert_sanity(self):
        for trace in self._traces:
            assert trace.route is not None
            assert trace.coordinate_route is not None
            assert trace.departure_time is not None

