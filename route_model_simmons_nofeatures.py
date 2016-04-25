

from route_model_simmons import RouteModelSimmons


class RouteModelSimmonsNoFeatures(RouteModelSimmons):

    def learn_routes(self, routes, _):
        for r in routes:
            route, features = self._split_route(r)
            if not len(route):
                continue
            self._learn_route(route, route[-1], ())

    def predict_route(self, partial_route, features):
        return RouteModelSimmons.predict_route(self, partial_route, ())

    def predict_arrival(self, partial_route, features):
        return RouteModelSimmons.predict_arrival(self, partial_route, ())

