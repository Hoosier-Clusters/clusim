""" useful functions for plotting """


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
    spines_to_remove = ['top', 'right', 'left', 'bottom']  # 'left', 'bottom']
    for spine in spines_to_remove:
        axs.spines[spine].set_visible(False)

    # Get rid of ticks. The position of the numbers is informative enough of
    # the position of the value.
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')

    axs.set_xticks([])
    axs.set_yticks([])
    # ax.set(aspect = 1)

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

        >>> import clusim.clugen as clugen
        >>> from clusim.plotutils import print_clustering
        >>> clu = clugen.make_equal_clustering(n_elements = 9, n_clusters = 3)
        >>> print_clustering(clu)
    """
    print('|'.join("".join(map(str, loe)) for loe
                   in clustering.clu2elm_dict.values()))
