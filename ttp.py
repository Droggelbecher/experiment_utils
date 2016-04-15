#!/usr/bin/env python

import csv

def extract_roadids(ttp_filename):
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
    roadids = []
    coordinates = []
    departure_time = None
    arrival_time = None
    road_ids_to_endpoints = {}

    f = open(ttp_filename, 'r')
    csv_reader = csv.reader(f)

    prev_road_id = None
    prev_lat = None
    prev_lon = None
    road_id = None


    for row in csv_reader:
        if row[1] == '129':
            try:
                t = float(row[-3])
            except ValueError as e:
                pass
            else:
                if departure_time is None:
                    departure_time = t
                arrival_time = t


        elif row[1] == '131':
            try:
                road_id = int(row[-4])
                lon = float(row[2])
                lat = float(row[3])
            except ValueError as e:
               pass

            else:
                if road_id != prev_road_id:

                    if prev_road_id is not None:
                        roadids.append(prev_road_id)
                        coordinates.append((lat, lon))
                        # road id changed -> set end coordinate of prev
                        road_ids_to_endpoints[prev_road_id] = (
                                road_ids_to_endpoints[prev_road_id][0],
                                (prev_lat, prev_lon)
                                )

                    # -> set start coordinate of new
                    road_ids_to_endpoints[road_id] = ( (lat, lon), None )

                road_ids_to_endpoints[road_id] = (
                    road_ids_to_endpoints[road_id][0],
                    (lat, lon)
                    )

                prev_road_id = road_id
                prev_lat = lat
                prev_lon = lon

    if road_id is not None:
        # road id changed -> set end coordinate of prev
        road_ids_to_endpoints[prev_road_id] = (
                road_ids_to_endpoints[prev_road_id][0],
                (lat, lon)
                )

    return {
            'roadids': roadids,
            'coordinates': coordinates,
            'departure_time': departure_time,
            'arrival_time': arrival_time,
            'road_ids_to_endpoints': road_ids_to_endpoints,
            }



