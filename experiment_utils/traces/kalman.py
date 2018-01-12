
def df_zip(*args):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], columns=('a', 'b', 'c'))
    >>> it = df_zip(df, df.iloc[1:])
    >>> left, right = next(it)
    >>> list(left[['a', 'b', 'c']])
    [1, 2, 3]
    >>> list(right[['a', 'b', 'c']])
    [4, 5, 6]
    """
    import numpy as np
    import pandas as pd

    widths = [len(x.columns) + 1 for x in args] # + 1 for the index which iloc seems to count as a column
    cumwidths = np.cumsum([0] + widths)
    df = pd.concat([x.reset_index() for x in args], axis = 1, join = 'inner')

    for idx, row in df.iterrows():
        yield tuple([row.iloc[w0:w1] for w0, w1 in zip(cumwidths, cumwidths[1:])])

def kalman_filter(path, dt):

    # A shitty kalman filter implementation
    # using a horribly simplified model of car motion and GNSS

    import numpy as np
    from math import sin, cos
    import pandas as pd

    if not len(path):
        # empty path, not much to do here
        return pd.DataFrame(columns = ('t', 'lat', 'lon'))

    print(path)
    import sys
    sys.stdout.flush()

    p = path.iloc[0]

    # Initial MLE estimate
    # x = [lat, lon, vlat, vlon]
    x = np.array([p['lat'], p['lon'], 0, 0])

    # Initial covariance matrix
    P = np.array([
        # lat  lon  vlat  vlon
        [ 1.0, 0.0, 0.0,  0.0 ], # lat
        [ 0.0, 1.0, 0.0,  0.0 ], # lon
        [ 0.0, 0.0, 1.0,  0.0 ], # vlat
        [ 0.0, 0.0, 0.0,  1.0 ], # vlon
        ])

    # GPS positon variance in meters
    gps_err_m = 1.0

    # GPS velocity estimation variance (m/s) per seconds between fixes
    gps_err_ms2 = 0.5

    # Simple kinematic matrix
    # (currently treats car like a billard ball)
    F = np.array([
        [ 1.0, 0.0,  dt, 0.0 ],
        [ 0.0, 1.0, 0.0,  dt ],
        [ 0.0, 0.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ]
        ])

    # Noise from untracked influence in each kinematic update step
    # just assume that all values will get worse
    # by some amount
    # (actually bad idea: lat/lon distances are far from constant across the globe!)
    # Numbers currently pulled out of my arse
    Q = np.array([
        [ 0.00001 * dt, 0.0, 0.0, 0.0 ],
        [ 0.0, 0.00001 * dt, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.00001 * dt, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.00001 * dt ],
        ])

    df = pd.DataFrame(columns = ('t', 'lat', 'lon'))

    LAT = 0
    LON = 1

    # Measurements
    #     TODO:
    #         f2 = pd.concat([df.reset_index(), df.iloc[1:].reset_index()], axis = 1, join = 'inner')
    for z_prev, z in df_zip(path, path.iloc[1:]):

        # Intermediate kinematic updates
        for t in np.arange(z_prev['t'] + dt, z['t'], dt):
            # Kinematic update
            x = x @ F
            P = F @ P @ F.T + Q
            assert not np.isnan(t)
            assert not np.isnan(x[LAT])
            assert not np.isnan(x[LON])
            df = df.append({'t': t, 'lat': x[LAT], 'lon': x[LON]}, ignore_index = True) # lat, lon

        dt_measurement = z['t'] - z_prev['t']

        m_per_deg_lat = 111132.954 - 559.822 * cos( 2 * x[LAT] ) + 1.175 * cos( 4 * x[LAT]);
        m_per_deg_lon = 111132.954 * cos ( x[LAT]);

        # Sensor noise
        R = np.array([
            [ gps_err_m / m_per_deg_lat, 0.0, 0.0, 0.0 ],
            [ 0.0, gps_err_m / m_per_deg_lon, 0.0, 0.0 ],
            # m/s^2 * dt * deg/m = deg/s
            [ 0.0, 0.0, gps_err_ms2 * dt_measurement / m_per_deg_lat, 0.0 ],
            [ 0.0, 0.0, 0.0, gps_err_ms2 * dt_measurement / m_per_deg_lon ],
            ])

        z_np = z[['lat', 'lon', 'vlat', 'vlon']].values

        # Measurement update
        # K is the kalman gain
        #print('P=', P)
        #print('R=', R)
        K = P @ np.linalg.inv(P + R)
        #print('K=', K)
        #print('z=', z)
        #print('z_np=', z_np)
        #print('x=', x)
        #print('z_np - x = ', z_np - x)
        #print('K @ (z_np - x) = ', K[:2, :2] @ (z_np - x)[:2])
        # [:2] --> just work on lat/lon not on vlat/vlon (as in general that info is not in the trace)
        x[:2] += (K[:2, :2] @ (z_np - x)[:2])
        #print('x=', x)
        P -= K @ P
        #df = df.append(x)
        assert not np.isnan(z['t'])
        assert not np.isnan(x[LAT])
        assert not np.isnan(x[LON])
        df = df.append({'t': z['t'], 'lat': x[LAT], 'lon': x[LON]}, ignore_index = True) # lat, lon

    df.set_index('t', inplace=True)
    return df

