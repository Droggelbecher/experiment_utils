

def plot_trace_points(df, **kws):
    from ggplot import geom_point, geom_line, geom_path
    return geom_point(data = df, **kws)

def plot_trace_path(df, **kws):
    from ggplot import geom_point, geom_line, geom_path
    return geom_path(data = df, **kws)


