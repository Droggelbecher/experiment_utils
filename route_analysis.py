#!/usr/bin/env python

import numpy as np
import scipy
import scipy.sparse
import csv
import sys

from collections import Counter

csv.field_size_limit(sys.maxsize)

def roadid_covariance_matrix(routes):
    all_road_ids = set()
    for route in routes:
        all_road_ids.update(route)
    all_road_ids = np.array(sorted(all_road_ids))


    co_occurence = Counter()
    for route_a in routes:
        for road_id_i in route_a:
            for road_id_j in route_a:
                if road_id_i <= road_id_j:
                    co_occurence[road_id_i, road_id_j] += 1



    covariance = np.zeros(shape = (len(all_road_ids), len(all_road_ids)), dtype=np.float_)

    L = float(len(routes))
    LL = L ** 2.0
    means = np.zeros(shape=(len(all_road_ids)))

    for i, road_id_i in enumerate(all_road_ids):
        means[i] = co_occurence[road_id_i, road_id_i] / L

        for j, road_id_j in enumerate(all_road_ids[i:]):
            c = co_occurence[road_id_i, road_id_j] / L
            c -= co_occurence[road_id_i, road_id_i] * co_occurence[road_id_j, road_id_j] / LL

            covariance[i, i + j] = c
            covariance[i + j, i] = c

            #if j != 0 and covariance[i, i + j] != 0:
                #print (i, j, covariance[i, i + j])

    return (all_road_ids, means, covariance)

if __name__ == '__main__':
    ids, cov = roadid_covariance_matrix([sys.argv[1]])
    print cov
    print np.where(cov != 0)



