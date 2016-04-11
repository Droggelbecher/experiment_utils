#!/usr/bin/env python

import csv


def extract_roadids(ttp_filename):
    """
    Requires a map-matched ttp 0.2
    """
    road_ids = set()

    f = open(ttp_filename, 'r')
    csv_reader = csv.reader(f)

    for row in csv_reader:
        if row[1] == '131':
            try:
                road_id = int(row[-4])
                road_ids.add(road_id)
            except ValueError:
                pass

    return road_ids



