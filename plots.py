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
    f, axs = plt.subplots(features - 1, features - 1, squeeze=False, sharex=True, sharey=True)

    for row in range(1, features):
        for column in range(features - 1):
            if column < row:
                axs[row - 1, column].scatter(a[:,column], a[:,row], alpha = 0.5, c=np.arange(a.shape[0]))
                axs[row - 1, column].set_xlabel(str(column))
                axs[row - 1, column].set_ylabel(str(row))
                axs[row - 1, column].get_xaxis().set_ticks([])
                axs[row - 1, column].get_yaxis().set_ticks([])
            else:
                axs[row - 1, column].axis('off')


    #plt.autoscale()
    plt.xlim((np.min(a), np.max(a)))
    plt.ylim((np.min(a), np.max(a)))

    f.set_size_inches((2*features, 2*features))
    f.savefig(filename, dpi=100)

