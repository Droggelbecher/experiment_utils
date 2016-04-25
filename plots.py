#!/usr/bin/env python

import numpy as np

#import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def all_relations(a, filename):
    """
    a: np array, rows for data, columns for features/axes
    """

    plt.clf()
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

def relation(xs, ys, filename):
    a = np.array([xs, ys])
    cov = np.cov(a)
    plt.clf()
    plt.scatter(xs, ys)
    plt.savefig(filename, dpi=100)

def cdfs(a, filename):
    """
    a: [ [ ... ], [ ... ], ... ]
    """

    plt.clf()
    plt.cla()
    for d, c in zip(a, plt.cm.Set1(np.linspace(0, 1, len(a)))):
        values = d['values']
        avg = np.sum(values)/len(values)
        median = np.median(values)

        def indices_to_counts(xs, indices):
            counts = indices[1:] - indices[:-1]
            l = len(xs) - sum(indices)
            return xs, np.hstack((counts, np.array([l])))

        #xs, counts = np.unique(values, return_counts = True)
        xs, counts = indices_to_counts( *np.unique(values, return_index = True) )
        ys = np.cumsum(counts)

        plt.plot(xs, ys, c=c, label=d['label'])
        plt.axvline(x = avg, color=c, linestyle=':') 
        plt.axvline(x = median, color=c, linestyle='--') 

    plt.legend(loc='best', prop={'size': 8})
    plt.savefig(filename, dpi=100)

def boxplots(xs, yss, filename):
    plt.clf()
    plt.cla()
    #fig, axes = plt.subplots(1, 1)

    plt.boxplot(yss, vert = True)
    plt.setp(plt.axes(), xticklabels=xs)
    
    plt.autoscale()

    plt.savefig(filename, dpi=100)

def multi_boxplots(xs, ysss, filename):
    fig, axes = plt.subplots(1, 1)

    for yss, c in zip(ysss, plt.cm.Set1(np.linspace(0, 1, len(ysss)))):
        axes.boxplot(yss, vert = True, boxprops={'color': c})

    plt.setp(axes, xticklabels=xs)
    fig.savefig(filename, dpi=100)
