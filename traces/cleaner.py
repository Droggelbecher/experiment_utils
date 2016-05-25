


def _transform_traces(traces):
    traces.assert_sanity()

    # Delete all empty/invalid entries
    to_delete = []
    for i, trace in enumerate(traces):
        if len(trace.route) == 0 or len(trace.coordinate_route) == 0:
            to_delete.append(i)

    for i in reversed(to_delete):
        del traces[i]

    traces.shuffle(lambda: 0.42)


    traces.assert_sanity()
    return traces



class Cleaner:

    def fit(self, X, y = None):
        return self

    def transform(self, traces):
        return _transform_traces(traces)
