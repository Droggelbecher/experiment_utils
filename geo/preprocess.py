

def smooth(trace, dt):
    """
    Simple smoothing & resampling of trace using splines.
    Not well-suited for noise or duplicates.

    trace: data frame with at least columns 't', 'lat', 'lon'
    dt: resampling time interval
    return: ???


    >>> import pandas as pd
    >>> df = pd.DataFrame(data = { 't': [1, 2, 3, 4, 5], 'lat': [10, 12, 10, 9, 8], 'lon': [100, 99, 98, 95, 97] })
    >>> smooth(df, 0.1)
    """

    import scipy.interpolate
    import pandas as pd
    import numpy as np

    start = trace.t.iloc[0]
    end = trace.t.iloc[-1]
    nt = start + np.linspace(0, end - start, (end - start) / dt + 1)


    r = pd.DataFrame(columns = ('t', 'lat', 'lon'))
    r['t'] = nt
    r['lat'] = scipy.interpolate.spline(trace.t, trace.lat, nt)
    r['lon'] = scipy.interpolate.spline(trace.t, trace.lon, nt)
    r.set_index('t', inplace=True)

    return r


def drop_duplicates(trace):
    """
    Remove entries that only differ in column 't', except for first.

    trace: data frame with at least column 't'
    """
    columns = set(trace.axes[1]) - set('t')
    return trace.drop_duplicates(subset = columns)




