# -*- coding: utf-8 -*-
"""
.. module:: clusimelement
    :synopsis: Element-centric Clustering Similarity

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import numpy as np

try:
    import igraph
except ImportError:
    pass

import scipy.sparse as spsparse

import collections
import itertools


def element_sim(clustering1, clustering2, alpha=0.9, r=1., r2=None, rescale_path_type='max'):
    """
    The element-centric clustering similarity.

    See :cite:`Gates2018element` for a detailed explaination of the measure.

    :param Clustering clustering1: The first Clustering

    :param Clustering clustering2: The second Clustering

    :param float alpha: The personalized page-rank return probability as a float in [0,1].

    :param float r1: The hierarchical scaling parameter for clustering1.

    :param float r2: The hierarchical scaling parameter for clustering2. This defaults to None
        forcing r2 = r1

    :param str rescale_path_type: rescale the hierarchical height by
        'max' : the maximum path from the root
        'min' : the minimum path form the root
        'linkage' : use the linkage distances in the clustering

    :param dict relabeled_elements: (optional)
        The elements maped to indices of the affinity matrix.

    :returns: The element-wise similarity between the two clusterings

    >>> import clusim.sim as sim
    >>> from clusim.clustering import Clustering
    >>> clustering1 = Clustering(elm2clu_dict={0:[0], 1:[0], 2:[0,1],
                                               3:[1], 4:[2], 5:[2]})
    >>> clustering2 = Clustering(elm2clu_dict={0:[0,2], 1:[0], 2:[0,1],
                                               3:[1], 4:[2], 5:[1,2]})
    >>> print(sim.element_sim(clustering1, clustering2, alpha=0.9))
    """

    result_tuple = element_sim_elscore(clustering1, clustering2, alpha=alpha,
                                       r=r, r2=r2,
                                       rescale_path_type=rescale_path_type)
    elementScores, relabeled_elements = result_tuple
    return np.mean(elementScores)


def element_sim_elscore(clustering1, clustering2, alpha=0.9, r=1., r2=None,
                        rescale_path_type='max', relabeled_elements=None):
    """
    The element-centric clustering similarity for each element.

    See :cite:`Gates2018element` for a detailed explaination of the measure.

    :param Clustering clustering1: The first Clustering

    :param Clustering clustering2: The second Clustering

    :param float alpha: The personalized page-rank return probability as a float in [0,1].

    :param float r1: The hierarchical scaling parameter for clustering1.

    :param float r2: The hierarchical scaling parameter for clustering2.  This defaults to None
        forcing r2 = r1

    :param str rescale_path_type: rescale the hierarchical height by:
        'max' : the maximum path from the root
        'min' : the minimum path form the root
        'linkage' : use the linkage distances in the clustering

    :param dict relabeled_elements: (optional)
        The elements maped to indices of the affinity matrix.

    :returns: The element-centric similarity between the two clusterings for each element as a 1d numpy array

    :returns: a dict mapping each element to its index of the elementScores array.

    >>> import clusim.sim as sim
    >>> from clusim.clustering import Clustering
    >>> clustering1 = Clustering(elm2clu_dict={0:[0], 1:[0], 2:[0,1],
                                               3:[1], 4:[2], 5:[2]})
    >>> clustering2 = Clustering(elm2clu_dict={0:[0,2], 1:[0], 2:[0,1],
                                               3:[1], 4:[2], 5:[1,2]})
    >>> elementScores, relabeled_elements = sim.element_sim_elseq(clustering1,
                                                              clustering2,
                                                              alpha = 0.9)
    >>> print(elementScores)
    """

    # the rows and columns of the affinity matrix correspond to relabeled
    # elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering1.elements)

    if r2 is None:
        r2 = r

    # make the two affinity matrices
    clu_affinity_matrix1 = make_affinity_matrix(clustering1, alpha=alpha, r=r,
                                                rescale_path_type=rescale_path_type,
                                                relabeled_elements=relabeled_elements)
    clu_affinity_matrix2 = make_affinity_matrix(clustering2, alpha=alpha, r=r2,
                                                rescale_path_type=rescale_path_type,
                                                relabeled_elements=relabeled_elements)

    # use the corrected L1 similarity
    nodeScores = cL1(clu_affinity_matrix1, clu_affinity_matrix2, alpha=alpha)

    return nodeScores, relabeled_elements


def relabel_objects(object_list):
    if np.all([type(i)==int for i in object_list]):
        relabeled_elements = {obj: iobj for iobj, obj in enumerate(sorted(object_list))}
    else:
        relabeled_elements = {obj: iobj for iobj, obj in enumerate(sorted(object_list,
                                                        key=lambda v: str(v)))}
    return relabeled_elements


def cL1(x, y, alpha):
    """
    The normalized similarity value based on the L1 probabilty metric
    corrected for the guaranteed overlap in probability between the two
    vectors, alpha.

    See :cite:`Gates2018element` for a detailed explaination of the need to correct
    the L1 metric.

    :param 2d-numpy-array x:
        The first list of probability vectors

    :param 2d-numpy-array y:
        The second list of probability vectors

    :param float alpha:
        The guaranteed overlap in probability between the two vectors in [0,1].

    :returns:
        The 1d numpy array of L1 similarities between the affinity matrices x and y
    """
    return 1.0 - 1.0/(2.0 * alpha) * np.sum(np.abs(x - y), axis=1)


def make_affinity_matrix(clustering, alpha=0.9, r=1., rescale_path_type='max',
                         relabeled_elements=None):
    """
    The element-centric clustering similarity affinity matrix for a
    clustering.  This function automatically determines the most efficient method
    to calculate the affinity matrix.

    See :cite:`Gates2018element` for a detailed explaination of the affinity matrix.


    :param Clustering clustering: The clustering

    :param float alpha: The personalized page-rank return probability.

    :param dict relabeled_elements: (optional)
        The elements maped to indices of the affinity matrix.


    :returns:
        The element-centric affinity representation of the clustering as a 2d numpy array

    >>> import clusim.sim as sim
    >>> from clusim.clustering import Clustering
    >>> clustering1 = Clustering(elm2clu_dict={0:[0], 1:[0], 2:[1], 3:[1],
                                               4:[2], 5:[2]})
    >>> pprmatrix = sim.make_affinity_matrix(clustering1, alpha=0.9)
    >>> print(pprmatrix)
    >>> clustering2 = Clustering(elm2clu_dict={0:[0], 1:[0], 2:[0,1], 3:[1],
                                               4:[2], 5:[2]})
    >>> pprmatrix2 = sim.make_affinity_matrix(clustering2, alpha=0.9)
    >>> print(pprmatrix2)
    """

    # the rows and cols of the affinity matrix correspond to relabeled elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering.elements)

    # check if the clustering is a partition
    if clustering.is_disjoint and not clustering.is_hierarchical:
        pprscore = ppr_partition(clustering=clustering, alpha=alpha,
                                 relabeled_elements=relabeled_elements)

    # otherwise we have to create the cielg and numberically solve for the
    # personalize page-rank distribution
    else:
        cielg = make_cielg(clustering=clustering, r=r,
                             rescale_path_type=rescale_path_type,
                             relabeled_elements=relabeled_elements)
        pprscore = numerical_ppr_scores(cielg, clustering, alpha=alpha,
                                        relabeled_elements=relabeled_elements)

    return(pprscore)


def ppr_partition(clustering, alpha=0.9, relabeled_elements=None):
    """
    The element-centric clustering similarity affinity matrix for a partition found analytically.

    :param Clustering clustering: The Clustering

    :param float alpha: The personalized page-rank return probability as a float in [0,1].

    :param dict relabeled_elements: (optional)
        The elements maped to indices of the affinity matrix.


    :returns: 2d numpy array
        The element-centric affinity representation of the clustering

    >>> import clusim.sim as sim
    >>> from clusim.clustering import Clustering
    >>> elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]}
    >>> clustering1 = Clustering(elm2clu_dict=elm2clu_dict)
    >>> pprmatrix = sim.ppr_partition(clustering1, alpha=0.9)
    >>> print(pprmatrix)
    """

    # the rows and columns of the affinity matrix correspond to relabeled
    # elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering.elements)

    ppr = np.zeros((clustering.n_elements, clustering.n_elements))
    for clustername, clusterlist in clustering.clu2elm_dict.items():
        Csize = len(clusterlist)
        clusterlist = [relabeled_elements[v] for v in clusterlist]
        ppr_result = alpha/Csize * np.ones((Csize, Csize)) +\
            np.eye(Csize) * (1.0 - alpha)
        ppr[[[v] for v in clusterlist], clusterlist] = ppr_result

    return ppr


def make_cielg(clustering, r=1.0, rescale_path_type='max',
                relabeled_elements=None):
    """
    Create the cluster-induced element graph for a Clustering.

    :param Clustering clustering: The clustering

    :param float r: The hierarchical scaling parameter.

    :param str rescale_path_type: rescale the hierarchical height by:
        'max' : the maximum path from the root
        'min' : the minimum path form the root
        'linkage' : use the linkage distances in the clustering

    :param dict relabeled_elements: (optional)
        The elements maped to indices of the affinity matrix.

    :returns:
        The cluster-induced element graph for a Clustering as an igraph.WeightedGraph

    >>> import clusim.sim as sim
    >>> from clusim.clustering import Clustering
    >>> elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]}
    >>> clustering1 = Clustering(elm2clu_dict=elm2clu_dict)
    >>> pprmatrix = sim.make_cielg(clustering1, r = 1.0)
    >>> print(pprmatrix)
    """

    # the rows and columns of the affinity matrix correspond to relabeled
    # elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering.elements)

    # the hierarchical weight function
    if clustering.is_hierarchical:
        cluster_height = clustering.hier_graph.rescale(
                             rescale_path_type=rescale_path_type)

        def weight_function(c): return np.exp(r * (cluster_height[c]))
        clu2elm_dict = clustering.hier_clusdict()
    else:
        def weight_function(c): return 1.0
        clu2elm_dict = clustering.clu2elm_dict

    relabeled_clusters = relabel_objects(clustering.clusters)

    edge_seq = []
    edge_weight_seq = []
    for c, element_list in clu2elm_dict.items():
        cstrength = weight_function(c)
        for el in element_list:
            edge_seq.append([relabeled_elements[el], relabeled_clusters[c]])
            edge_weight_seq.append(cstrength)

    edge_seq = np.array(edge_seq)

    bipartite_adj = spsparse.coo_matrix((edge_weight_seq,
                                         (edge_seq[:, 0], edge_seq[:, 1])),
                                        shape=(clustering.n_elements,
                                               clustering.n_clusters))
    proj1 = bipartite_adj / bipartite_adj.sum(axis=1)
    proj2 = bipartite_adj / bipartite_adj.sum(axis=0)
    projected_adj = proj1.dot(proj2.T)
    cielg = igraph.Graph.Weighted_Adjacency(projected_adj.tolist(),
                                             mode=igraph.ADJ_DIRECTED,
                                             attr="weight", loops=True)
    return cielg


def find_groups_in_cluster(clustervs, elementgroupList):
    """
    A utility function to find vertices with the same cluster
    memberships.

    :param igraph.vertex clustervs: an igraph vertex instance

    :param list elementgroupList: a list containing the vertices to group


    :returns:
        a list-of-lists containing the groupings of the vertices
    """
    clustervertex = set([v for v in clustervs])
    return [vg for vg in elementgroupList if len(set(vg) & clustervertex) > 0]


def numerical_ppr_scores(cielg, clustering, alpha=0.9,
                         relabeled_elements=None):
    """
    The element-centric clustering similarity affinity matrix for a partition.

    :param igraph.WeightedGraph cielg: cielg : An igraph Weighted Graph representation of the cluster-induced element graph

    :param Clustering clustering: The Clustering

    :param float alpha: The personalized page-rank return probability as a float in [0,1].

    :param dict relabeled_elements: (optional) dict
        The elements maped to indices of the affinity matrix.


    :returns: 2d numpy array
        The element-centric affinity representation of the clustering
    """

    collect_regulargroups = collections.defaultdict(list)
    for e, cl in clustering.elm2clu_dict.items():
        collect_regulargroups[tuple(sorted(cl))].append(relabeled_elements[e])
    elementgroupList = collect_regulargroups.values()

    ppr_scores = np.zeros((cielg.vcount(), cielg.vcount()))

    # we have to calculate the ppr for each connected component
    for cluster in cielg.components():
        clustergraph = cielg.subgraph(cluster)
        cc_ppr_scores = np.zeros((clustergraph.vcount(),
                                  clustergraph.vcount()))

        for elementgroup in find_groups_in_cluster(cluster, elementgroupList):
            # we only have to solve for the ppr distribution once per group
            vertex = clustergraph.vs[cluster.index(elementgroup[0])]
            cc_ppr_scores[vertex.index] = clustergraph.personalized_pagerank(
                directed=True, weights="weight", damping=(alpha),
                reset_vertices=vertex, implementation='prpack')

            # the other vertices in the group are permutations of that solution
            for v2 in elementgroup[1:]:
                v2 = cluster.index(v2)
                cc_ppr_scores[v2] = cc_ppr_scores[vertex.index]
                cc_ppr_scores[v2, vertex.index] = cc_ppr_scores[vertex.index, v2]
                cc_ppr_scores[v2, v2] = cc_ppr_scores[vertex.index, vertex.index]

        ppr_scores[[[v] for v in cluster], cluster] = cc_ppr_scores

    return ppr_scores


def element_sim_matrix(clustering_list, alpha=0.9, r=1.,
                       rescale_path_type='max'):

    relabeled_elements = relabel_objects(clustering_list[0].elements)

    affinity_matrix_list = [make_affinity_matrix(clustering, alpha=alpha, r=r,
                            rescale_path_type=rescale_path_type,
                            relabeled_elements=relabeled_elements)
                            for clustering in clustering_list]

    Nclusterings = len(clustering_list)
    sim_matrix = np.zeros(int(Nclusterings * (Nclusterings - 1) / 2))
    icompare = 0
    for iclustering, jclustering in itertools.combinations(range(Nclusterings),
                                                           2):
        sim_matrix[icompare] = np.mean(cL1(affinity_matrix_list[iclustering],
                                       affinity_matrix_list[jclustering],
                                       alpha))
        icompare += 1

    return sim_matrix
