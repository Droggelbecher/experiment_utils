#!/usr/bin/env python

import csv
import sys
csv.field_size_limit(sys.maxsize)

from cache import cached

class ImplausibleRoute(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

@cached()
def extract_roadids(ttp):
    """
    Requires a map-matched ttp 0.2

    returns:
        roadids: list of traversed road ids, in order, without duplicates
        coordinates: list of road segment entry coordinates, one for each roadids entry
            eg.: [ (52.123, 9.123), (52.125, 9.124), ...  ]
        departure_time: start timestamp (unix time)
        arrival_time: end timestamp
        road_ids_to_endpoints: dict road_id => ((lat_enter, lon_enter), (lat_leave, lon_leave))

    """

    # Max. seconds between positioned+mapmatched gps fixes
    MAX_DELTA_T = 100.0

    roadids = []
    coordinates = []
    departure_time = None
    arrival_time = None
    road_ids_to_endpoints = {}

    csv_reader = csv.reader(ttp.splitlines())

    prev_road_id = None
    prev_lat = None
    prev_lon = None
    road_id = None

    t = None
    for row in csv_reader:
        if row[1] == '129':
            if row[-3] == 'INVALID':
                continue

            t_prev = t
            t = float(row[-3])

            if t_prev is not None:
                if not (0.0 <= t - t_prev <= MAX_DELTA_T):
                    raise ImplausibleRoute('suspicious time skip! t_prev={} t={}'.format(t_prev, t))

            if departure_time is None:
                departure_time = t
            arrival_time = t


        elif row[1] == '131':
            if row[-4] == 'INVALID':
                continue

            #try:
            road_id = int(row[-4])
            lon = float(row[2])
            lat = float(row[3])
            #except ValueError as e:
               #pass

            #else:
            if road_id != prev_road_id:

                if prev_road_id is not None:
                    # We are now done with $prev_road_id
                    roadids.append(prev_road_id)
                    coordinates.append((lat, lon))
                    # road id changed -> set end coordinate of prev
                    road_ids_to_endpoints[prev_road_id] = (
                            road_ids_to_endpoints[prev_road_id][0],
                            (prev_lat, prev_lon)
                            )

                # We are now starting with $road_id
                # -> set start coordinate of new
                road_ids_to_endpoints[road_id] = ( (lat, lon), None )

                #road_ids_to_endpoints[road_id] = (
                    #road_ids_to_endpoints[road_id][0],
                    #(lat, lon)
                    #)

                prev_road_id = road_id
            prev_lat = lat
            prev_lon = lon

    if road_id is not None:
        roadids.append(road_id)
        coordinates.append((lat, lon))

        road_ids_to_endpoints[road_id] = (
                road_ids_to_endpoints[road_id][0],
                (lat, lon)
                )

    return {
            'roadids': roadids,
            'coordinates': coordinates,
            'departure_time': departure_time,
            'arrival_time': arrival_time,
            'road_ids_to_endpoints': road_ids_to_endpoints,
            }



