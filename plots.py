#!/usr/bin/env python

import sys
import numpy as np
import math
import logging
import re

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size = 12)

import iterutils

def fill_between_steps(ax, x, y1, y2=0, step_where='pre', **kwargs):
    ''' fill between a step plot and 

    source: https://github.com/matplotlib/matplotlib/issues/643/

    Parameters
    ----------
    ax : Axes
       The axes to draw to

    x : array-like
        Array/vector of index values.

    y1 : array-like or float
        Array/vector of values to be filled under.
    y2 : array-Like or float, optional
        Array/vector or bottom values for filled area. Default is 0.

    step_where : {'pre', 'post', 'mid'}
        where the step happens, same meanings as for `step`

    **kwargs will be passed to the matplotlib fill_between() function.

    Returns
    -------
    ret : PolyCollection
       The added artist

    '''
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays 
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array 

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    if step_where == 'pre':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == 'post':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = np.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    xx, yy1, yy2 = steps

    # now to the plotting part:
    return ax.fill_between(xx, yy1, y2=yy2, **kwargs)




def all_relations(a, filename, labels = None):
    """
    a: np array, rows for data, columns for features/axes
    """

    plt.clf()
    features = a.shape[1]
    f, axs = plt.subplots(features - 1, features - 1, squeeze=False) #, sharex=True, sharey=True)

    if labels is None:
        labels = [str(x) for x in range(features)]

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
    f.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(f)

def relation(xs, ys, filename, xlabel=None, ylabel=None, pointlabels = [],
        xlim = None, xlog = False):
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

    if xlim is not None:
        ax.set_xlim(xlim)

    if xlog:
        ax.set_xscale('log')

    if pointlabels:
        for x, y, l in zip(xsl, ysl, pointlabels):
            ax.text(x, y, l, transform = trans_offset, fontsize=8)

    ax.grid(True)

    fig.savefig(filename, dpi=100)
    plt.close(fig)

def relations(xss, yss, filename, xlabel=None, ylabel=None, pointlabels = [],
        xlim = None, labels = []):
    xss = list(xss)
    yss = list(yss)

    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)

    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x = 0.05, y = 0.1, units='inches')

    if len(labels) < len(xss):
        labels += [''] * (len(xss) - len(labels))

    for xsl, ysl, c, label in zip(xss, yss, plt.cm.Dark2(np.linspace(0, 1, len(xss))), labels):
        ax.scatter(xsl, ysl, c = c, alpha=0.5, label=label)
        if pointlabels:
            for x, y, l in zip(xsl, ysl, pointlabels):
                ax.text(x, y, l, transform = trans_offset, fontsize=8)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)


    plt.legend(loc='upper center', prop={'size': 10}, bbox_to_anchor=(0.5,1.1), ncol=int(math.ceil(len(xss)/2.0)), fancybox=True)

    ax.grid(True)

    fig.savefig(filename, dpi=100)
    plt.close(fig)



def curves(xss, yss,
           filename,
           yss_min = [], yss_max = [],
           labels = [],
           invert_x = False,
           xlabel = None, ylabel = None,
           xoffsets = None,
           yoffsets = None,
           xlog = False, ylog = False,
           xlim = None, ylim = None,
           step = False,
           closeups_x = [],
           linestyle = '-',
           legendrows = 2,
           cm = plt.cm.Dark2,
           minmax_style = 'filled',
           markers = '^os',
           markeverys = [],
           grid = False,
           figsize = None,
          ):

    xlim_ = xlim
    filename_ = filename
    if len(labels) < len(xss):
        labels += [''] * (len(xss) - len(labels))

    if len(yss_min) < len(yss):
        yss_min += [None] * (len(yss) - len(yss_min))

    if len(yss_max) < len(yss):
        yss_max += [None] * (len(yss) - len(yss_max))

    if xoffsets is None:
        xoffsets = 0

    if yoffsets is None:
        yoffsets = 0

    if len(markeverys) < len(xss):
        markeverys += [ None ] * (len(xss) - len(markeverys))

    if isinstance(linestyle, str):
        linestyle = [linestyle]



    for i, xlim in enumerate([xlim_] + closeups_x):
        fig, ax = plt.subplots(1, 1, figsize = figsize)

        if i == 0:
            filename = filename_
        else:
            m = re.match(r'(.*)(\.[^.]*)$', filename_)
            if m is None:
                filename + filename_ + '_closeup{}'.format(i) + '.pdf'
            else:
                filename = m.groups()[0] + '_closeup{}'.format(i) + m.groups()[1]

        if xlog:
            ax.set_xscale('log', basex=xlog)

        if ylog:
            ax.set_yscale('log', basey=ylog)

        if invert_x:
            ax.invert_xaxis()

        ax.yaxis.grid(grid, which = 'major', linestyle='-', color='.8')
        ax.yaxis.grid(grid, which = 'minor', linestyle=':', color='.8')

        offsx = 0 
        offsy = 0
        for xs, ys, ys_min, ys_max, c, label, marker, markevery, ls in zip(
            xss,
            yss,
            yss_min,
            yss_max,
            cm(np.linspace(0, 1, len(yss))),
            labels,
            iterutils.repeat_to(markers, len(xss)),
            markeverys,
            iterutils.repeat_to(linestyle, len(xss)),
        ):
            ys = np.array(ys)
            xs = np.array(xs)

            if ys_min is not None:
                ys_min = np.array(ys_min)

            if ys_max is not None:
                ys_max = np.array(ys_max)
            

            aox = np.full((len(xs), ), offsx)
            aoy = np.full((len(ys), ), offsy)

            offsx += xoffsets
            offsy += yoffsets

            if step:
                ax.step(xs + aox, ys + aoy, ls, where = 'post', c = c, label = label, markeredgewidth = 0.0)

            else:
                if minmax_style == 'error' and ys_min is not None and ys_max is not None:
                    ax.plot(xs + aox, ys + aoy, ls, marker = marker, c = c, label = label, markeredgewidth = 0.0, markevery = markevery, markersize=8)
                    ax.errorbar(
                        xs + aox, ys + aoy,
                        yerr = [-(ys_min - ys + aoy), ys_max - ys + aoy],
                        alpha = 0.8,
                        #lolims = True,
                        #uplims = True,
                        capsize = 1.5,
                        linestyle = '',
                        linewidth = 0,
                        elinewidth = 0.5,
                        c = c,
                        markeredgecolor = c,
                        color = c,
                    )
                else:
                    ax.plot(xs + aox, ys + aoy, ls, marker = marker, c = c, label = label, markeredgewidth = 0.0)

            if ys_min is not None and ys_max is not None and minmax_style == 'filled':
                if step:
                    fill_between_steps(ax, xs + aox, ys_min + aoy, ys_max + aoy, color = c, alpha = 0.1, linestyle = '--', step_where = 'post')

                else:
                    ax.fill_between(xs + aox, ys_min + aoy, ys_max + aoy, color = c, alpha = 0.1)

            # "Special" markers

            #saox = np.full((len(sxs), ), offsx)
            #saoy = np.full((len(sys), ), offsy)
            #ax.plot(sxs + saox, sys + saoy, '', marker = '*', c = c)


        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        logging.debug('len(yss)={} legendrows={}'.format(len(yss), legendrows))
        try:
            ax.legend(
                loc='upper center',
                prop={'size': 8},
                bbox_to_anchor=(0.5, 1.2),
                ncol=int(math.ceil(len(yss) / legendrows)),
                fancybox=True,
                frameon=False
            )
        except IndexError:
            pass
        logging.debug('creating {}'.format(filename))
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)



def percentile_curves(xss, ysss, filename, low = 10, high = 90, **kws):

    yss = [ np.array([np.median(s[i:]) for i in range(len(s))]) for s in ysss ]
    yss_min = [ np.array([np.percentile(s[i:], q = low) for i in range(len(s))]) for s in ysss ]
    yss_max = [ np.array([np.percentile(s[i:], q = high) for i in range(len(s))]) for s in ysss ]

    curves(
            xss = xss,
            yss = yss,
            filename = filename,
            yss_min = yss_min,
            yss_max = yss_max,
            **kws
            )



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
    plt.savefig(filename, dpi=100, bbox_inches='tight')

def boxplots(xs, yss, filename):
    plt.clf()
    plt.cla()
    #fig, axes = plt.subplots(1, 1)

    plt.boxplot(yss, vert = True)
    plt.setp(plt.axes(), xticklabels=xs)
    
    plt.autoscale()

    plt.savefig(filename, dpi=100, bbox_inches='tight')

def multi_boxplots(xs, ysss, filename, ylim = (-0.05, 1.05), labels = [], toplabels = [], points = True, xlabel = '', ylabel = '', legendrows = 2, cm = plt.cm.Dark2):
    """
    ysss = [
            [ [ x x x x ], ... ],
            [ [ x x x x ], ... ],
            ....
        ]

    toplabels = [
        [ 'color1box1', 'color1box2', ... ],
        [ 'color2box1', 'color2box2', ... ],
        ]

    labels = ['color1', 'color2', 'color3', ...]

    innermost list = data for 1 box
    list of boxlists = data row (same color)
    """

    fig, axes = plt.subplots(1, 1, figsize=(14, 7))

    if ylim is None:
        elems = iterutils.flatten(ysss)
        if len(elems):
            ylim = (min(elems), max(elems))

            delta = (ylim[1] - ylim[0]) * 0.05
            ylim = (ylim[0] - delta, ylim[1] + delta)

        else:
            ylim = (0, 1)




    top = ylim[1] + (ylim[1] - ylim[0]) * 0.1

    k = len(ysss) + 1
    dummylines = []
    maxlen = 0
    if len(labels) < len(ysss):
        labels += [str(i) for i in range(len(ysss) - len(labels))]

    if not toplabels:
        toplabels = [ [] ] * len(ysss)

    for yss, c, offset, tlables in zip(
            ysss,
            #plt.cm.Dark2(np.linspace(0, 1, len(ysss))),
            cm(np.linspace(0, 1, len(ysss))),
            range(1, 1 + len(ysss)),
            toplabels
            ):
        maxlen = max(maxlen, len(yss))

        # each yss is a data set and gets one color
        ps = range(offset, len(yss) * k + offset, k)

        if len(yss):
            bp = axes.boxplot(yss,
                    vert = True,
                    boxprops={'color': c},
                    widths = 0.6,
                    positions = ps)

            # Style the box
            for key, v in bp.items():
                for e in v:
                    plt.setp(e, color = c)

        # dummy line for legend
        h, = plt.plot([1, 1], c = c, color = c, linestyle='-')
        dummylines.append(h)


        for ys, p in zip(yss, ps):
            rxs = np.random.normal(p, 0.06, size = len(ys))
            plt.plot(rxs, ys, '.', c = c, alpha = 0.5)


        if tlables:
            for p, label in zip(ps, tlables):
                axes.text(p, top * 0.95, label, horizontalalignment='center', size=12, weight='bold', color = c)

    axes.yaxis.grid(True)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)


    if len(dummylines) > 1:
        plt.legend(dummylines, labels, loc='upper center', prop={'size': 10}, bbox_to_anchor=(0.5,1.1), ncol=int(math.ceil(len(dummylines)/legendrows)), fancybox=True)
    for l in dummylines:
        l.set_visible(False)

    plt.setp(axes, xticks = np.arange(k/2.0, k/2.0 + maxlen * k, k), xticklabels=xs)
    axes.set_xlim((0, maxlen * k))

    if ylim is not None:
        ylim = (ylim[0], top)
        axes.set_ylim(ylim)

    fig.savefig(filename, dpi=100, bbox_inches = 'tight')
    plt.close(fig)


def matrix(a, filename):
    plt.clf()

    aspect_ratio = a.shape[1] / float(a.shape[0])
    print(4 * aspect_ratio, 4)

    fig, ax = plt.subplots(dpi=100)
    cs = ax.matshow(a)
    fig.colorbar(cs)
    fig.savefig(filename, dpi=100, bbox_inches = 'tight')
    plt.close(fig)


def kde1d(xs_plot, estimators, filename, xss_data = [], labels = None,
        xlabel = None, xlim = None, lines = [], grid = True):

    plt.clf()
    fig, ax = plt.subplots(dpi=100)

    if labels is None:
        labels = ['KDE {}'.format(x) for x in range(len(estimators))]

    cm = plt.cm.Set2(np.linspace(0, 1, len(estimators)))
    alpha = 0.5
    offs = 0.01
    spread = 0.005
    
    for i, (kde, label, xs_data, c) in enumerate(zip(estimators, labels, xss_data, cm)):
        log_dens = kde.score_samples(xs_plot.reshape(len(xs_plot), 1))
        ax.fill(xs_plot, np.exp(log_dens), '-', label = label, alpha = alpha, fc
                = c)

        ax.legend(loc='best', prop={'size': 8})
        ax.plot(xs_data, -(i+1) * offs + spread/2 - spread * np.random.random(len(xs_data)), 'o', c=c,
                alpha=alpha)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    for line in lines:
        ax.axvline(c = 'grey', linestyle = '-', x = line)

    ax.grid(grid)
    fig.savefig(filename, dpi=100, bbox_inches='tight')

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

