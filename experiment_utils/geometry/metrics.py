
import math
from experiment_utils.warnings import deprecated

@deprecated
def geo(pos1, pos2):
    """
    Returns distance in meters between 2 lat/lon coordinates.

    pos1: Source coordinate (expected to have [lat] & [lon])
    pos2: Target coordinate (expected to have [lat] & [lon])
    """
    lat1 = pos1.lat
    lon1 = pos1.lon
    lat2 = pos2.lat
    lon2 = pos2.lon

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
