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

def cdfs(a, filename):
    """
    a: [ [ ... ], [ ... ], ... ]
    """

    for d, c in zip(a, plt.cm.Set1(np.linspace(0, 1, len(a)))):
        xs = d['values']


        avg = np.sum(xs)/len(xs)
        median = np.median(xs)

        xs, counts = np.unique(xs, return_counts = True)
        ys = np.cumsum(counts)

        plt.plot(xs, ys, c=c, label=d['label'])
        plt.axvline(x = avg, color=c, label=d['label'] + ' avg', linestyle=':') 
        plt.axvline(x = median, color=c, label=d['label'] + ' median', linestyle='--') 

    plt.legend(loc='best', prop={'size': 8})
    plt.savefig(filename, dpi=100)


