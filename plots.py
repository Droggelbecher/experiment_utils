#!/usr/bin/env python

import numpy as np

#import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def all_relations(a, filename):
    """
    a: np array, rows for data, columns for features/axes
    """

    features = a.shape[1]
    f, axs = plt.subplots(features, features, squeeze=False, sharex=True, sharey=True)

    for row in range(features):
        for column in range(features):
            if column <= row:
                axs[row, column].scatter(a[:,column], a[:,row], alpha = 0.5, c=np.arange(a.shape[0]))
            else:
                axs[row, column].axis('off')

    #plt.autoscale()
    plt.xlim((np.min(a), np.max(a)))
    plt.ylim((np.min(a), np.max(a)))

    f.set_size_inches((2*features, 2*features))
    f.savefig(filename, dpi=100)

