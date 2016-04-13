#!/usr/bin/env python

import csv


def extract_roadids(ttp_filename):
    """
    Requires a map-matched ttp 0.2

    """
    road_ids_to_endpoints = {}
    path = []

    f = open(ttp_filename, 'r')
    csv_reader = csv.reader(f)

    prev_road_id = None
    prev_lat = None
    prev_lon = None
    for row in csv_reader:
        if row[1] == '131':
            try:
                road_id = int(row[-4])
                lon = float(row[2])
                lat = float(row[3])
            except ValueError as e:
                print(e)

            else:
                path.append((lat, lon))

                if road_id != prev_road_id:
                    if prev_road_id is not None:
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
    return path, road_ids_to_endpoints



