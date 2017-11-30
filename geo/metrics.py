


def geo_distance(p1, p2):
    return abs(p1 - p2) # TODO

def dtw_distance(path1, path2):

    """

    >>> geo_distance = lambda a, b: a - b

    >>> path1 = [1, 2, 3, 4, 5]
    >>> path2 = [1, 2, 2, 3, 3, 4, 5]
    >>> dtw_distance(path1, path2)
    0
    """

    import numpy as np

    n = len(path1)
    m = len(path2)

    dtw = np.zeros((n + 1, m + 1))
    dtw[:, 0] = np.inf
    dtw[0, :] = np.inf
    dtw[0, 0] = 0

    for i, p1 in enumerate(path1):
        for j, p2 in enumerate(path2):
            cost = geo_distance(p1, p2)
            dtw[i + 1, j + 1] = cost + min(
                dtw[i, j + 1],  # insertion
                dtw[i + 1, j],  # deletion
                dtw[i, j] # match
                )

    return dtw[n, m]


