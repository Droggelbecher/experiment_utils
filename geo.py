
import math

def distance(lat1, lon1, lat2, lon2):
    """
    Returns distance in meters between 2 lat/lon points
    """
    degrees = math.pi / 180.0
    phi1 = (90.0 - lat1) * degrees
    phi2 = (90.0 - lat2) * degrees

    theta1 = lon1 * degrees
    theta2 = lon2 * degrees

    c = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) + math.cos(phi1) * math.cos(phi2))
    return 6373000.0 * math.acos(c)

