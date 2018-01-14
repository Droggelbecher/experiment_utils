
def smooth_regress(path, dt, order):
    """
    path: data frame with at least columns 't', 'lat', 'lon'
    dt: resampling time interval
    order: order of the polynomial to use
    return: interpolated path
    """
    import pandas as pd
    import numpy as np
    from numpy import polyfit, poly1d

    start = path.t.iloc[0]
    end = path.t.iloc[-1]
    # new ts sequence
    nt = start + np.linspace(0, end - start, (end - start) / dt + 1)

    avg_t = np.mean(path['t'])
    avg_lat = np.mean(path['lat'])
    avg_lon = np.mean(path['lon'])

    lat = np.poly1d(np.polyfit(path['t'] - avg_t, path['lat'] - avg_lat, order))
    lon = np.poly1d(np.polyfit(path['t'] - avg_t, path['lon'] - avg_lon, order))

    r = pd.DataFrame(columns = ('t', 'lat', 'lon'))
    r['t'] = nt
    r['lat'] = list(map(lat, nt - avg_t))
    r['lon'] = list(map(lon, nt - avg_t))

    # Repair path
    r['lat'] += avg_lat
    r['lon'] += avg_lon
    r.set_index('t', inplace=True)

    return r


def smooth_spline(path, dt):
    """
    Simple smoothing & resampling of trace using splines.
    Not well-suited for noise or duplicates.

    path: data frame with at least columns 't', 'lat', 'lon'
    dt: resampling time interval
    return: interpolated path

    TODO: allow to work without 't' attribute (and use distance instead),
          as that is the usecase for map arcs

    >>> import pandas as pd
    >>> df = pd.DataFrame(data = { 't': [1, 2, 3, 4, 5], 'lat': [10, 12, 10, 9, 8], 'lon': [100, 99, 98, 95, 97] })
    >>> smooth(df, 0.1)
    """

    import scipy.interpolate
    import pandas as pd
    import numpy as np

    start = path.t.iloc[0]
    end = path.t.iloc[-1]
    # new ts sequence
    nt = start + np.linspace(0, end - start, (end - start) / dt + 1)

    r = pd.DataFrame(columns = ('t', 'lat', 'lon'))
    r['t'] = nt
    r['lat'] = scipy.interpolate.spline(path.t, path.lat, nt)
    r['lon'] = scipy.interpolate.spline(path.t, path.lon, nt)
    r.set_index('t', inplace=True)

    return r


def drop_duplicates(trace):
    """
    Remove entries that only differ in column 't', except for first.

    trace: data frame with at least column 't'
    """
    columns = set(trace.axes[1]) - set('t')
    return trace.drop_duplicates(subset = columns)





