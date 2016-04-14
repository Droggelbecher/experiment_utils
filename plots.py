#!/usr/bin/env python

#import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def all_relations(a):
    """
    a: np array, rows for data, columns for features/axes
    """

    features = a.shape[1]

    #fig = plt.figure(figsize = (2*features, 2*features), dpi=100)
    #fig.set_size_inches(1*features, 1*features)

    f, axs = plt.subplots(features, features, squeeze=False, sharex=True, sharey=True)

    i = 1
    for row in range(features):
        for column in range(features):
            axs[row, column].scatter(a[:,column], a[:,row])

    f.set_size_inches((2*features, 2*features))
    f.savefig('/tmp/all_relations.pdf', dpi=100)

