""" useful functions for plotting """
from six import iteritems

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def cm2inch(*tupl):
    """ convert cm to inches """
    cm_per_inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/cm_per_inch for i in tupl[0])
    else:
        return tuple(i/cm_per_inch for i in tupl)

def blank_axis(axs):
    """ remove spines, ticks, and axis lines from the axis object """

    # Remove top and right axes lines ("spines")
    spines_to_remove = ['top', 'right', 'left', 'bottom']#, 'left', 'bottom']
    for spine in spines_to_remove:
        axs.spines[spine].set_visible(False)

    # Get rid of ticks. The position of the numbers is informative enough of
    # the position of the value.
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')

    axs.set_xticks([])
    axs.set_yticks([])
    #ax.set(aspect = 1)

    for xlabel in axs.axes.get_xticklabels():
        xlabel.set_visible(False)

    for ylabel in axs.axes.get_yticklabels():
        ylabel.set_visible(False)

    return axs

def print_clustering(clustering):
    """
        A function to print a clustering where clusters are seperated by '|'.

        Parameters
        ----------
        clustering : Clustering
            The clustering to print

        >>> import clusim
        >>> clu = make_equal_clustering(n_elements = 9, n_clusters = 3)
        >>> print_clustering(clu)
    """
    print('|'.join("".join(map(str, loe)) for loe in clustering.clus2elm_dict.values()))

def draw_small_clustering(clustering, axs=None, params=None):
    """ given a mem_dict and an axis object, draw the clustering """

    # processing the parameters.
    if params is None:
        params = {}
    w_padding = params.get('w_padding', 0.05)
    fontsize = params.get('fontsize', 10)
    cmap = params.get('cmap', 'jet')
    alpha = params.get('alpha', 0.3)
    xlim = params.get('xlim', (-0.07, 1))
    ylim = params.get('xlim', (-0.1, 0.1))
    boxstyle = params.get('boxstyle', mpatches.BoxStyle("Round", pad=0.02))
    xmin = 0.0 + w_padding
    xmax = 1.0 - w_padding
    xspacing = (xmax - xmin)/float(clustering.number_of_elements()) 

    # create ax object if there is none provided.
    if axs is None:
        _, axs = plt.subplots(1, 1, figsize=(10, 1))
    axs = blank_axis(axs)
    axs.set_xlim(*xlim)
    axs.set_ylim(*ylim)

    patches = []
    for _, elms in sorted(iteritems(clustering.clu_dict),
                          key=lambda x: int(x[0].strip('.'))):
        cstart = xmin + min(elms) * xspacing #- 0.95 * w_padding
        clength = (max(elms) - min(elms)) * xspacing
        fancybox = mpatches.FancyBboxPatch([cstart, -0.05],
                                           clength,
                                           0.1,
                                           boxstyle=boxstyle)
        patches.append(fancybox)

    colors = np.linspace(0, 1, len(patches))
    collection = PatchCollection(patches, cmap=cmap, alpha=alpha)
    collection.set_array(np.array(colors))
    axs.add_collection(collection)

    for elm_idx, elm in enumerate(sorted(clustering.elements)):
        axs.text(xmin + elm_idx * xspacing,
                 0.0, str(elm), ha='center', va='center', fontsize=fontsize)
