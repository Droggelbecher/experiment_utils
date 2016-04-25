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
    xsl = list(xs)
    ysl = list(ys)
    a = np.array([xsl, ysl])
    cor = np.corrcoef(a, rowvar=False)
    plt.axes().arrow(0, 0, 1, cor[0, 1])
    plt.clf()
    plt.scatter(xsl, ysl)
    plt.savefig(filename, dpi=100)

def cdfs(a, filename):
    """
    a: [ [ ... ], [ ... ], ... ]
    """

    plt.clf()
    plt.cla()
    for i, (d, c) in enumerate(zip(a, plt.cm.Paired(np.linspace(0, 1, len(a))))):
        values = d['values']
        values = np.sort(values)
        avg = np.sum(values)/len(values)
        median = np.median(values)

        def indices_to_counts(xs, indices):
            counts = indices[1:] - indices[:-1]
            l = len(values) - sum(counts)
            return xs, np.hstack((counts, np.array([l])))

        #xs, counts = np.unique(values, return_counts = True)
        xs, counts = indices_to_counts( *np.unique(values, return_index = True) )
        ys = np.cumsum(counts)

        plt.plot(xs, ys, c=c, label=d['label'], linewidth=i/5+1)
        plt.axvline(x = avg, color=c, linestyle=':', linewidth=i/5+1) 
        plt.axvline(x = median, color=c, linestyle='--', linewidth=i/5+1) 

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


if __name__ == '__main__':

    a = [
        { 'label': 'foo00', 'values': np.random.rand(10) },
        { 'label': 'foo01', 'values': np.random.rand(10) },
        { 'label': 'foo02', 'values': np.random.rand(10) },
        { 'label': 'foo03', 'values': np.random.rand(10) },
        { 'label': 'foo04', 'values': np.random.rand(10) },
        { 'label': 'foo05', 'values': np.random.rand(10) },
        { 'label': 'foo06', 'values': np.random.rand(10) },
        { 'label': 'foo07', 'values': np.random.rand(10) },
        { 'label': 'foo08', 'values': np.random.rand(10) },
        { 'label': 'foo09', 'values': np.random.rand(10) },
        { 'label': 'foo10', 'values': np.random.rand(10) },
        { 'label': 'foo11', 'values': np.random.rand(10) },
        { 'label': 'foo12', 'values': np.random.rand(10) },
        { 'label': 'foo13', 'values': np.random.rand(10) },
        { 'label': 'foo14', 'values': np.random.rand(10) },
        { 'label': 'foo15', 'values': np.random.rand(10) },
        { 'label': 'foo16', 'values': np.random.rand(10) },
        { 'label': 'foo17', 'values': np.random.rand(10) },
        { 'label': 'foo18', 'values': np.random.rand(10) },
        { 'label': 'foo19', 'values': np.random.rand(10) },
        
        ]

    cdfs(a, '/tmp/test.pdf')
