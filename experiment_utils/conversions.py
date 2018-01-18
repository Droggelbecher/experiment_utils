
from typing import Any, Sequence
import numpy as np

def df_to_rows(df):
    return tuple(x for x in df.itertuples(index = False))

def make_like(p: Any, v: Sequence[Any]):
    """
    Convert v to be somewhat like p.
    See doctests for examples of supported types.
    Tries to avoid copying, so return value might or might not share structure with v.

    >>> import numpy as np
    >>> import pandas as pd

    >>> l = [ (1, 2, 3), (4, 5, 6) ]
    >>> a = np.array(l)
    >>> df = pd.DataFrame(a, columns=('t', 'x', 'y'))
    >>> r = list(df.itertuples(index = False))

    >>> make_like(a, l)
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> make_like(df, l)
       t  x  y
    0  1  2  3
    1  4  5  6
    >>> make_like(r, l)
    [Pandas(t=1, x=2, y=3), Pandas(t=4, x=5, y=6)]

    >>> make_like(l, a)
    [(1, 2, 3), (4, 5, 6)]
    >>> make_like(df, a)
       t  x  y
    0  1  2  3
    1  4  5  6
    >>> make_like(r, a)
    [Pandas(t=1, x=2, y=3), Pandas(t=4, x=5, y=6)]

    >>> make_like(l, df)
    [(1, 2, 3), (4, 5, 6)]
    >>> make_like(a, df)
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> make_like(r, df)
    [Pandas(t=1, x=2, y=3), Pandas(t=4, x=5, y=6)]

    """
    import pandas as pd

    if (
        (type(v) is type(p) is pd.DataFrame) or
        (type(v) is type(p) is np.array) or
        (type(v) is type(p) and type(v[0]) is type(p[0]))
        ):
        return v

    if isinstance(p, pd.DataFrame):
        return type(p)(v, columns=p.columns, copy=False)

    if isinstance(p, np.ndarray):
        return np.array(v, copy=False)

    # Make sure after this point v is something iterable
    if isinstance(v, pd.DataFrame):
        v = v.itertuples(index = False)

    if isinstance(p[0], tuple):
        try:
            # namedtuple constructor
            return type(p)([type(p[0])(*x) for x in v])
        except TypeError:
            # actual tuple
            return type(p)([type(p[0])(x) for x in v])

    return type(p)(type(p[0])(x) for x in v)

