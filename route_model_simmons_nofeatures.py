

import numpy as np

from route_model_simmons import RouteModelSimmons

class RouteModelSimmonsNoFeatures(RouteModelSimmons):

    def learn_routes(self, routes, features, _):
        return RouteModelSimmons.learn_routes(self, routes, np.zeros((len(routes), 0)), _)

    def predict_route(self, partial_route, features):
        return RouteModelSimmons.predict_route(self, partial_route, ())

    def predict_arrival(self, partial_route, features):
        return RouteModelSimmons.predict_arrival(self, partial_route, ())

