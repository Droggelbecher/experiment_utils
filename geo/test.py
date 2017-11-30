
from plot import plot_trace
from preprocess import smooth

import pandas as pd

from ggplot import ggplot, aes, geom_point, geom_line

if __name__ == '__main__':
    df = pd.DataFrame(data = { 't': [1, 2, 3, 4, 5], 'lat': [10, 12, 10, 9, 8], 'lon': [100, 99, 98, 95, 97] })
    df2 = smooth(df, 0.1)

    p = ggplot(aes(x='lat', y='lon'), data=pd.DataFrame(columns=('lat', 'lon'), data={}))

    p += plot_trace(df, color='red')
    print(df2)
    p += plot_trace(df2)

    p.save('test.png')


