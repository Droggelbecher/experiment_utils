#!/usr/bin/env python

import numpy as np

#import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

def all_relations(a, filename, labels = None):
    """
    a: np array, rows for data, columns for features/axes
    """

    plt.clf()
    features = a.shape[1]
    f, axs = plt.subplots(features - 1, features - 1, squeeze=False) #, sharex=True, sharey=True)

    if labels is None:
        labels = [str(x) for x in range(features)]

    #plt.xlim((np.min(a), np.max(a)))
    #plt.ylim((np.min(a), np.max(a)))
    for row in range(1, features):
        for column in range(features - 1):
            if column < row:
                axs[row - 1, column].scatter(a[:,column], a[:,row], alpha = 0.5, c=np.arange(a.shape[0]))

                if column > 0:
                    axs[row - 1, column].get_yaxis().set_ticks([])
                else:
                    axs[row - 1, column].set_ylabel(labels[row])

                if row < features - 1: 
                    axs[row - 1, column].get_xaxis().set_ticks([])
                else:
                    axs[row - 1, column].set_xlabel(labels[column])

                if len(a[:,column]):
                    axs[row - 1, column].set_xlim((np.min(a[:,column]), np.max(a[:,column])))

                if len(a[:,row]):
                    axs[row - 1, column].set_ylim((np.min(a[:,row]), np.max(a[:,row])))
            else:
                axs[row - 1, column].axis('off')


    #plt.autoscale()

    f.set_size_inches((2*features, 2*features))
    f.savefig(filename, dpi=100)
    plt.close(f)

def relation(xs, ys, filename, xlabel=None, ylabel=None, pointlabels = []):
    xsl = list(xs)
    ysl = list(ys)

    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x = 0.05, y = 0.1, units='inches')

    ax.scatter(xsl, ysl, c = plt.cm.Set1(np.linspace(0, 1, len(xsl))), alpha=0.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if pointlabels:
        for x, y, l in zip(xsl, ysl, pointlabels):
            ax.text(x, y, l, transform = trans_offset, fontsize=8)

    fig.savefig(filename, dpi=100)
    plt.close(fig)


def curves(xs, yss, filename, labels = [], invert_x = False, xlabel = None, ylabel = None):
    fig, ax = plt.subplots(1, 1)

    ax.set_xscale('log', basex=2)

    if invert_x:
        ax.invert_xaxis()

    for ys, c, label in zip(yss, plt.cm.Set1(np.linspace(0, 1, len(yss))), labels):
        ax.plot(xs, ys, '-', c = c, label = label)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.legend(loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True)
    fig.savefig(filename)
    plt.close(fig)




def cdfs(a, filename):
    """
    a: [ { 'label': 'foo', 'values': [ ... ] }, ... ]
    """

    plt.clf()
    plt.cla()
    for i, (d, c) in enumerate(zip(a, plt.cm.Paired(np.linspace(0, 1, len(a))))):
        values = d['values']

        if len(values) == 0:
            values = [0]

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

def multi_boxplots(xs, ysss, filename, ylim = None, labels = []):
    """
    ysss = [ [ [ x x x x ], ... ], .... ]
    """

    fig, axes = plt.subplots(1, 1, figsize=(14, 7))

    k = len(ysss) + 1
    dummylines = []
    maxlen = 0
    if len(labels) < len(ysss):
        labels += [str(i) for i in range(len(ysss) - len(labels))]

    for yss, c, offset in zip(
            ysss,
            plt.cm.Set1(np.linspace(0, 1, len(ysss))),
            range(1, 1 + len(ysss))):
        maxlen = max(maxlen, len(yss))

        # each yss is a data set and gets one color
        ps = range(offset, len(yss) * k + offset, k)
        bp = axes.boxplot(yss,
                vert = True,
                boxprops={'color': c},
                widths = 0.6,
                positions = ps)

        for key, v in bp.items():
            for e in v:
                plt.setp(e, color = c)

        # dummy line for legend
        h, = plt.plot([1, 1], 'r-', c = c, color = c, linestyle='-')
        dummylines.append(h)

    plt.legend(dummylines, labels, loc='upper center', prop={'size': 8}, bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True)
    for l in dummylines:
        l.set_visible(False)

    plt.setp(axes, xticks = np.arange(k/2.0, k/2.0 + maxlen * k, k), xticklabels=xs)
    axes.set_xlim((0, maxlen * k))
    if ylim is not None:
        axes.set_ylim(ylim)
    fig.savefig(filename, dpi=100)
    plt.close(fig)


def matrix(a, filename):
    plt.clf()
    fig, ax = plt.subplots(figsize=(4, .5), dpi=100)
    ax.matshow(a, cmap=plt.cm.gray)
    fig.savefig(filename, dpi=1000)
    plt.close(fig)


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

    xs = [1.23, 4.56, 7.89]
    ysss = [
            [ [1, 1, 3, 8, 15], [6, 8, 7], [9, 0, 6, 7] ],
            [ [2, 2, 3, 8, 6], [10, 10, 10, 6], [1,2,3,4,5] ],
           ]
    multi_boxplots(xs, ysss, '/tmp/multi_boxplots.pdf', labels=['foo', 'bar'])

