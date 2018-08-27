# -*- coding: utf-8 -*-
"""
.. module:: clugen
    :synopsis: A set of functions to generate Clusterings and random Clusterings

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import numpy as np
import mpmath
import copy

from clusim.clustering import Clustering
from clusim.dag import Dendrogram


def make_equal_clustering(n_elements, n_clusters):
    """
    This function creates a random clustering with equally sized clusters.
    If n_elements % n_clusters != 0, cluster sizes will differ by one
    element.

    :param int n_elements:
        The number of elements

    :param int n_clusters:
        The number of clusters

    :returns:
        The new clustering with equally sized clusters.

    >>> import clusim.clugen as clugen
    >>> from clusim.clustering import print_clustering
    >>> clu = clugen.make_equal_clustering(n_elements = 9, n_clusters = 3)
    >>> print_clustering(clu)
    """
    new_elm2clu_dict = {el: [el % n_clusters] for el in range(n_elements)}
    new_clustering = Clustering(new_elm2clu_dict)
    return new_clustering


def make_random_clustering(n_elements=1, n_clusters=1, clu_size_seq=[1, 2],
                           random_model='all', tol=1.0e-15):
    """
    This function creates a random clustering according to one of three
    random models. It is a wrapper around the specific functions for each random model.

    :param int n_elements:
        The number of elements

    :param int n_clusters:
        The number of clusters

    :param str random_mode:
        The random model to use:

        'all' : uniform distrubtion over the set of all clusterings of
                n_elements

        'num' : uniform distrubtion over the set of all clusterings of
                n_elements in n_clusters

        'perm' : the Permutaiton Model

    :param float tol: optional
        The tolerance used by the algorithm for 'all' clusterings

    :returns:
        The new clustering.

    >>> import clusim.clugen as clugen
    >>> from clusim.clustering import print_clustering
    >>> clu = clugen.make_random_clustering(n_elements = 9, n_clusters = 3,
                                     random_model = 'num')
    >>> print_clustering(clu)
    """
    if random_model in ['all', 'all1']:
        new_clustering = generate_random_partition_all(n_elements=n_elements,
                                                       tol=tol)

    elif random_model in ['num', 'num1']:
        new_clustering = generate_random_partition_num(n_elements=n_elements,
                                                       n_clusters=n_clusters)

    elif random_model in ['perm']:
        new_clustering = generate_random_partition_perm(clu_size_seq)
    return new_clustering


def make_singleton_clustering(n_elements):
    """
    This function creates a clustering with each element in its own
    cluster.

    :param int n_elements:
        The number of elements

    :returns:
        The new clustering.

    >>> import import clusim.clugen as clugen
    >>> from clusim.clustering import print_clustering
    >>> clu = clugen.make_singleton_clustering(n_elements = 9)
    >>> print_clustering(clu)
    """
    new_clsutering = make_regular_clustering(n_elements=n_elements,
                                             n_clusters=n_elements)
    return new_clsutering


def make_random_dendrogram(n_elements):
    """
    This function creates a random Hierarchical Clustering.

    :param int n_elements The number of elements

    :returns:
        The new clustering.

    """
    dendro_graph = Dendrogram()
    dendro_graph.make_random_dendrogram_aglomerative(N=n_elements)
    return HierClustering(clu2elm_dict={e: set([e])
                                        for e in dendro_graph.leaves()},
                          hier_graph=dendro_graph)


def shuffle_memberships(clustering, percent=1.0):
    """
    This function creates a new clustering by shuffling the element
    memberships from the original clustering.

    :param Clustering clustering: The original clustering.

    :param float percent: optional (default 1.0)
        The fractional percentage (between 0.0 and 1.0) of the elements to
        shuffle.

    :returns: The new clustering.

    >>> import clusim.clugen as clugen
    >>> from clusim.clustering import print_clustering
    >>> orig_clu = clugen.make_random_clustering(n_elements = 9, n_clusters = 3,
                                          random_model = 'num')
    >>> print_clustering(orig_clu)
    >>> shuffle_clu = clugen.shuffle_memberships(orig_clu, percent = 0.5)
    >>> print_clustering(shuffle_clu)
    """
    el_to_shuffle = np.random.choice(clustering.elements,
                                     int(percent * clustering.n_elements),
                                     replace=False)
    shuffled_el = np.random.permutation(el_to_shuffle)
    newkeys = dict(zip(el_to_shuffle, shuffled_el))

    new_elm2clu_dict = copy.deepcopy(clustering.elm2clu_dict)
    for el in shuffled_el:
        new_elm2clu_dict[el] = clustering.elm2clu_dict[newkeys[el]]

    if clustering.is_hierarchical:
        new_clustering = HierClustering(elm2clu_dict=new_elm2clu_dict,
                                        hier_graph=copy.deepcopy(clustering.hiergraph))
    else:
        new_clustering = Clustering(elm2clu_dict=new_elm2clu_dict)
    return new_clustering


def shuffle_memberships_pa(clustering, n_steps=1, constant_num_clusters=True):
    """
        This function creates a new clustering by shuffling the element
        memberships from the original clustering according to the preferential
        attachment model.

        See :cite:`Gates2017impact` for a detailed explaination of the preferential
        attachment model.

        :param Clustering clustering: The original clustering.

        :param int n_steps: optional (default 1)
            The number of times to run the preferential attachment algorithm.

        :param Boolean constant_num_clusters: optional (default True)
            Reject a shuffling move if it leaves a cluster with no elements.
            Set to True to keep the number of clusters constant.

        :returns:
            The new clustering with shuffled memberships.

        >>> import clusim.clugen as clugen
        >>> from clusim.clustering import print_clustering
        >>> orig_clu = clugen.make_random_clustering(n_elements=9, n_clusters=3,
                                              random_model='num')
        >>> print_clustering(orig_clu)
        >>> shuffle_clu = clugen.shuffle_memberships_pa(orig_clu, n_steps=10,
                                                 constant_num_clusters=True)
        >>> print_clustering(shuffle_clu)
    """
    n_elements_norm = 1./float(clustering.n_elements)

    Nclusters = clustering.n_clusters

    cluster_list = clustering.to_cluster_list()
    cluster_size_prob = np.array(list(map(len, cluster_list))) * n_elements_norm
    clusternames = range(Nclusters)

    for istep in range(n_steps):
        from_cluster = np.random.choice(clusternames, p=cluster_size_prob)
        if cluster_size_prob[from_cluster] > 1.5*n_elements_norm or not constant_num_clusters:

            exchanged_element = np.random.choice(cluster_list[from_cluster], 1,
                                                 replace=False)[0]
            new_cluster = np.random.choice(clusternames, p=cluster_size_prob)

            if new_cluster != from_cluster:
                cluster_list[from_cluster].remove(exchanged_element)
                cluster_size_prob[from_cluster] -= n_elements_norm

                cluster_list[new_cluster].append(exchanged_element)
                cluster_size_prob[new_cluster] += n_elements_norm

    new_clustering = Clustering()
    new_clustering.from_cluster_list(cluster_list)

    return new_clustering


def generate_random_partition_perm(clu_size_seq):
    n_elements = sum(clu_size_seq)
    n_clusters = len(clu_size_seq)
    elm_list = np.random.permutation(np.arange(n_elements))
    clu_idx = np.hstack([[0], np.cumsum(clu_size_seq)])

    cluster_list = [elm_list[clu_idx[iclus]:clu_idx[iclus + 1]]
                    for iclus in range(n_clusters)]

    new_clustering = Clustering()
    new_clustering.from_cluster_list(cluster_list)
    return new_clustering


def _random_partition_num_iterator(n_elements, n_clusters):
    '''http://thousandfold.net/cz/2013/09/25/sampling-uniformly-from-the-set-of-partitions-in-a-fixed-number-of-nonempty-sets/'''

    assert n_clusters <= n_elements

    if n_elements == 1:
        current_partition = [[0]]

    else:
        stirling_prob = mpmath.stirling2(n_elements - 1, n_clusters - 1) / mpmath.stirling2(n_elements, n_clusters)

        if np.random.random() < stirling_prob:
            current_partition = _random_partition_num_iterator(n_elements=n_elements - 1, n_clusters=n_clusters - 1)
            current_partition.append([n_elements - 1])
        else:
            current_partition = _random_partition_num_iterator(n_elements=n_elements - 1, n_clusters=n_clusters)
            current_clu = np.random.randint(n_clusters)
            current_partition[current_clu].append(n_elements - 1)

    return current_partition


def generate_random_partition_num(n_elements, n_clusters):

    clu_list = _random_partition_num_iterator(n_elements, n_clusters)

    new_clustering = Clustering()
    new_clustering.from_cluster_list(clu_list)
    return new_clustering


all_partition_weight_dict = {}
def generate_random_partition_all(n_elements, tol=1.0e-15):
    """
        This function creates a random clustering according to the 'All'
        random model by uniformly selecting a clustering from the ensemble of all
        clusterings with n_elements.

        :param int n_elements:
            The number of elements

        :param float tol: (optional)
            The tolerance used by the algorithm to approximate the probability distrubtion

        :returns: The randomly genderated clustering.

        >>> import clusim.clugen as clugen
        >>> from clusim.clustering import print_clustering
        >>> clu = clugen.generate_random_partition_all(n_elements = 9)
        >>> print_clustering(clu)
    """

    if (n_elements, tol) in all_partition_weight_dict:
        weights = all_partition_weight_dict[(n_elements, tol)]
    else:
        weights = []
        u = 1
        b = mpmath.bell(n_elements)
        while sum(weights) < 1.0 - tol:
            weights.append(mpmath.power(u, n_elements)/(b * mpmath.e * mpmath.factorial(u)))
            u += 1
        all_partition_weight_dict[(n_elements, tol)] = weights

    K = np.random.choice(np.arange(1, len(weights) + 1), p=weights)
    colors = np.random.randint(K, size=n_elements)

    new_clustering = Clustering()
    new_clustering.from_membership_list(colors)
    return new_clustering


def enumerate_random_partition_num(n_elements, n_clusters):
    """
        A generator for every partition in 'Num', the ensemble of all clusterings
        with n_elements grouped into n_clusters, non-empty clusters.

        Based on the solution provided by Adeel Zafar Soomro: `a link`_.

        .. _a link: http://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions

        which was itself based on the algorithm from Knuth:
        (Algorithm U) is described by Knuth in the Art of Computer Programming,
        Volume 4, Fascicle 3B

        :param int n_elements:
            The number of elements

        :param int n_clusters:
            The number of clusters

        :returns:
            The new clustering as a cluster list.

        >>> import clusim.clugen as clugen
        >>> from clusim.clustering import print_clustering
        >>> for clu in clugen.clustering_ensemble_generator_num(n_elements=5, n_clusters=3):
        >>>     print_clustering(clu)
    """

    elm_list = range(n_elements)

    def visit(n, a):
        ps = [[] for i in range(n_clusters)]
        for j in range(n):
            ps[a[j + 1]].append(elm_list[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    if n_clusters == 1:
        return [[elm_list]]
    elif n_clusters == n_elements:
        return [[[i] for i in elm_list]]
    else:
        a = [0] * (n_elements + 1)
        for j in range(1, n_clusters + 1):
            a[n_elements - n_clusters + j] = j - 1
    return f(n_clusters, n_elements, 0, n_elements, a)
