


def geo(pos1, pos2):
    """
    Returns distance in meters between 2 lat/lon coordinates.

    pos1: Source coordinate (expected to have .lat & .lon)
    pos2: Target coordinate (expected to have .lat & .lon)
    """
    import math

    lat1 = float(pos1['lat'])
    lon1 = float(pos1['lon'])
    lat2 = float(pos2['lat'])
    lon2 = float(pos2['lon'])

    if lat1 == lat2 and lon1 == lon2:
        return 0.0


    degrees = math.pi / 180.0
    phi1 = (90.0 - lat1) * degrees
    phi2 = (90.0 - lat2) * degrees

    theta1 = lon1 * degrees
    theta2 = lon2 * degrees

    c = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) + math.cos(phi1) * math.cos(phi2))

    if c >= 1.0:
        acos = 0.0
    elif c <= -1.0:
        acos = math.pi
    else:
        acos = math.acos(c)

    return 6373000.0 * acos

def dtw(seq1, seq2, distance = geo):

    """
    Return the dynamic time warping distance between two sequnces,
    using the given element-wise distance metric.

    >>> seq1 = [1, 2, 3, 4, 5]
    >>> seq2 = [1, 2, 2, 3, 3, 4, 5]
    >>> dtw(seq1, seq2, distance = lambda a, b: abs(a - b))
    0.0
    """

    import numpy as np

    n = len(seq1)
    m = len(seq2)

    dtw = np.zeros((n + 1, m + 1))
    dtw[:, 0] = np.inf
    dtw[0, :] = np.inf
    dtw[0, 0] = 0

    for i, p1 in enumerate(seq1):
        for j, p2 in enumerate(seq2):
            cost = distance(p1, p2)
            dtw[i + 1, j + 1] = cost + min(
                dtw[i, j + 1],  # insertion
                dtw[i + 1, j],  # deletion
                dtw[i, j] # match
                )

    return dtw[n, m]


