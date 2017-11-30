

def plot_trace(df, **kws):
    from ggplot import geom_point, geom_line, geom_path

    p = geom_point(data = df, **kws)
    p += geom_path(data = df, **kws)
    return p


