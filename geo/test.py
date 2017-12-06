
from plot import plot_trace_points, plot_trace_path
from preprocess import smooth_spline, smooth_regress

import pandas as pd

from ggplot import ggplot, aes, geom_point, geom_line

if __name__ == '__main__':
    df = pd.DataFrame(data = { 't': [1, 2, 3, 4, 5], 'lat': [10, 12, 10, 9, 8], 'lon': [100, 99, 98, 95, 97] })
    df2 = smooth_spline(df, 0.1)
    df3 = smooth_regress(df, 0.1, 4)
    df4 = smooth_regress(df, 0.1, 3)

    p = ggplot(aes(x='lat', y='lon'), data=pd.DataFrame(columns=('lat', 'lon'), data={}))

    p += plot_trace_points(df, color='black')
    p += plot_trace_path(df2, color='red')
    p += plot_trace_path(df3, color='green')
    p += plot_trace_path(df4, color='blue')

    p.save('test.png')


