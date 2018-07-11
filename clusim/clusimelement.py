"""
Element-centric Clustering Similarity

TODO: implement the hierarchical version
"""
import numpy as np
import networkx as nx

try:
    import igraph
except ImportError:
    pass

import scipy.sparse as spsparse

import collections
import itertools

from clusim.clustering import Clustering

def element_sim_matrix(clustering_list, alpha = 0.9, r = 1., rescale_path_type = 'max'):

    relabeled_elements = relabel_objects(clustering_list[0].elements)

    affinity_matrix_list = [make_affinity_matrix(clustering, alpha = alpha, r = r, rescale_path_type = rescale_path_type, 
        relabeled_elements = relabeled_elements) for clustering in clustering_list]

    Nclusterings = len(clustering_list)
    sim_matrix = np.zeros( int(Nclusterings * (Nclusterings - 1) / 2) )
    icompare = 0
    for iclustering, jclustering in itertools.combinations(range(Nclusterings), 2):
        sim_matrix[icompare] = np.mean(cL1(affinity_matrix_list[iclustering], affinity_matrix_list[jclustering], alpha))
        icompare += 1

    return sim_matrix
    
        


def element_sim(clustering1, clustering2, alpha = 0.9, r = 1., r2 = None, rescale_path_type = 'max'):
    """
        The element-centric clustering similarity.

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        alpha : float
            The personalized page-rank return probability.

        Returns
        -------
        element_sim : float
            The element-wise similarity between the two clusterings

        >>> import clusim
        >>> clustering1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]})
        >>> clustering2 = Clustering(elm2clu_dict = {0:[0,2], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[1,2]})
        >>> print(element_sim(clustering1, clustering2, alpha = 0.9)) 
    """
    elementScores, relabeled_elements = element_sim_elscore(clustering1, clustering2, alpha = alpha, r = r, r2 = r2, rescale_path_type = rescale_path_type)
    return np.mean(elementScores)

def element_sim_elscore(clustering1, clustering2, alpha = 0.9, r = 1., r2 = None, rescale_path_type = 'max', relabeled_elements = None):
    """
        The element-centric clustering similarity for each element.

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        alpha : float
            The personalized page-rank return probability.

        relabeled_elements : dict, optional
            The elements maped to indices of the affinity matrix.

        Returns
        -------
        elementScores: numpy array
            The element-centric similarity between the two clusterings for each element

        relabeled_elements : dict
            The elements maped to indices of the elementScores array.

        >>> import clusim
        >>> clustering1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]})
        >>> clustering2 = Clustering(elm2clu_dict = {0:[0,2], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[1,2]})
        >>> elementScores, relabeled_elements = element_sim_elseq(clustering1, clustering2, alpha = 0.9)
        >>> print(elementScores) 
    """

    # the rows and columns of the affinity matrix correspond to relabeled elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering1.elements)

    if r2 is None:
        r2 = r

    # make the two affinity matrices
    clu_affinity_matrix1 = make_affinity_matrix(clustering1, alpha = alpha, r = r, 
        rescale_path_type = rescale_path_type, relabeled_elements = relabeled_elements)
    clu_affinity_matrix2 = make_affinity_matrix(clustering2, alpha = alpha, r = r2, 
        rescale_path_type = rescale_path_type, relabeled_elements = relabeled_elements) 

    # use the corrected L1 similarity
    nodeScores = cL1(clu_affinity_matrix1, clu_affinity_matrix2, alpha = alpha)
    
    return nodeScores, relabeled_elements

def relabel_objects(object_list):
    return {obj:iobj for iobj, obj in enumerate(sorted(object_list, key = lambda v: str(v))) }

def cL1(x, y, alpha):
    """
        The normalized similarity value based on the L1 probabilty metric corrected for the 
        guaranteed overlap in probability between the two vectors, alpha.

        Parameters
        ----------
        x : 2d numpy array
            The first list of probability vectors

        y : 2d numpy array
            The second list of probability vectors

        alpha : float
            The guaranteed overlap in probability between the two vectors.

        Returns
        -------
        cL1 : numpy array
            The list of L1 similarities between each pair of probability vectors
    """
    return 1.0 - 1.0/(2.0 * alpha) * np.sum(np.abs(x - y), axis = 1)


def make_affinity_matrix(clustering, alpha = 0.9, r = 1., rescale_path_type = 'max', relabeled_elements = None):
    """
        The element-centric clustering similarity affinity matrix for a clustering.

        Parameters
        ----------
        clustering : Clustering
            The clustering

        alpha : float
            The personalized page-rank return probability.

        relabeled_elements : dict, optional
            The elements maped to indices of the affinity matrix.

        Returns
        -------
        ppr: 2d numpy array
            The element-centric affinity representation of the clustering

        >>> import clusim
        >>> clustering1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
        >>> pprmatrix = make_affinity_matrix(clustering1, alpha = 0.9)
        >>> print(pprmatrix) 
        >>> clustering2 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]})
        >>> pprmatrix2 = make_affinity_matrix(clustering2, alpha = 0.9)
        >>> print(pprmatrix2) 
    """

    # the rows and columns of the affinity matrix correspond to relabeled elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering.elements)
    
    # check if the clustering is a partition
    if clustering.is_disjoint and not clustering.is_hierarchical:
        pprscore = ppr_partition(clustering = clustering, alpha = alpha, relabeled_elements = relabeled_elements)

    # otherwise we have to create the hctag and numberically solve for the personalize page-rank distribution
    else:
        phctag = make_phctag(clustering = clustering, r = r, rescale_path_type = rescale_path_type, relabeled_elements = relabeled_elements)
        pprscore = numerical_ppr_scores(phctag, clustering, alpha = alpha, relabeled_elements = relabeled_elements)
        
    return(pprscore)



def ppr_partition(clustering, alpha = 0.9, relabeled_elements = None):
    """
        The element-centric clustering similarity affinity matrix for a partition.

        Parameters
        ----------
        clustering : Clustering
            The clustering

        alpha : float
            The personalized page-rank return probability.

        relabeled_elements : dict, optional
            The elements maped to indices of the affinity matrix.

        Returns
        -------
        ppr: 2d numpy array
            The element-centric affinity representation of the clustering

        >>> import clusim
        >>> clustering1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
        >>> pprmatrix = ppr_partition(clustering1, alpha = 0.9)
        >>> print(pprmatrix) 
    """

    # the rows and columns of the affinity matrix correspond to relabeled elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering1.elements)

    ppr = np.zeros((clustering.n_elements, clustering.n_elements))
    for clustername, clusterlist in clustering.clu2elm_dict.items():
        Csize = len(clusterlist)
        clusterlist = [relabeled_elements[v] for v in clusterlist]
        ppr[[[v] for v in clusterlist] , clusterlist] = alpha/Csize * np.ones((Csize, Csize)) + np.eye(Csize) * (1.0 - alpha )

    return ppr


def make_phctag(clustering, r = 1.0, rescale_path_type = 'max', relabeled_elements = None):
    """
        The element-centric clustering similarity affinity matrix for a partition.

        Parameters
        ----------
        clustering : Clustering
            The clustering

        r : float
            The scaling parameter.

        relabeled_elements : dict, optional
            The elements maped to indices of the affinity matrix.

        Returns
        -------
        ppr: 2d numpy array
            The element-centric affinity representation of the clustering

        >>> import clusim
        >>> clustering1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
        >>> pprmatrix = ppr_partition(clustering1, alpha = 0.9)
        >>> print(pprmatrix) 
    """

    # the rows and columns of the affinity matrix correspond to relabeled elements
    if relabeled_elements is None:
        relabeled_elements = relabel_objects(clustering.elements)

    # the hierarchical weight function
    if clustering.is_hierarchical:
        cluster_height = clustering.hier_graph.rescale(rescale_path_type = rescale_path_type)
        weight_function = lambda c: np.exp( r * (cluster_height[c]))
        clu2elm_dict = clustering.hier_clusdict()
    else:
        weight_function = lambda c: 1.0
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
    
    bipartite_adj = spsparse.coo_matrix((edge_weight_seq, (edge_seq[:,0], edge_seq[:,1])), shape = (clustering.n_elements, clustering.n_clusters))
    proj1 = bipartite_adj / bipartite_adj.sum(axis = 1)
    proj2 = bipartite_adj / bipartite_adj.sum(axis = 0)
    projected_adj = proj1.dot(proj2.T)
    phctag = igraph.Graph.Weighted_Adjacency(projected_adj.tolist(), mode = igraph.ADJ_DIRECTED, attr = "weight", loops = True)
    return phctag


def find_groups_in_cluster(clustervs, elementgroupList):
    """
        A utility function to find vertices with all of the same cluster memberships.

        Parameters
        ----------
        clustervs : igraph vertex
            an igraph vertex instance

        elementgroupList : list of vertices
            a list containing the vertices to group


        Returns
        -------
        groupings: list of lists
            a list containing the groupings of the vertices
    """
    clustervertex = set([v for v in clustervs])
    return [vg for vg in elementgroupList if len(set(vg) & clustervertex) > 0]
    


def numerical_ppr_scores(phctag, clustering, alpha = 0.9, relabeled_elements = None):
    """
        The element-centric clustering similarity affinity matrix for a partition.

        Parameters
        ----------
        phctag : igraph Weighted Graph
            The projected HCTAG

        clustering : Clustering
            The clustering

        alpha : float
            The personalized page-rank return probability.

        relabeled_elements : dict, optional
            The elements maped to indices of the affinity matrix.

        Returns
        -------
        ppr: 2d numpy array
            The element-centric affinity representation of the clustering 
    """
    
    collect_regulargroups = collections.defaultdict(list)
    for e, cl in clustering.elm2clu_dict.items():
        collect_regulargroups[tuple(sorted(cl))].append(relabeled_elements[e])
    elementgroupList = collect_regulargroups.values()

    ppr_scores = np.zeros((phctag.vcount(), phctag.vcount()))

    # we have to calculate the ppr for each connected component
    for cluster in phctag.components():
        clustergraph = phctag.subgraph(cluster)
        cc_ppr_scores = np.zeros((clustergraph.vcount(), clustergraph.vcount()))


        for elementgroup in find_groups_in_cluster(cluster, elementgroupList):
            # we only have to solve for the ppr distribution once per group
            vertex = clustergraph.vs[cluster.index(elementgroup[0])]
            cc_ppr_scores[vertex.index] = clustergraph.personalized_pagerank(directed=True, weights = "weight", \
                    damping=(alpha), reset_vertices=vertex, implementation = 'prpack')

            # the other vertices in the group are permutations of that solution
            for v2 in elementgroup[1:]:
                v2 = cluster.index(v2)
                cc_ppr_scores[v2] = cc_ppr_scores[vertex.index]
                cc_ppr_scores[v2, vertex.index] = cc_ppr_scores[vertex.index, v2]
                cc_ppr_scores[v2, v2] = cc_ppr_scores[vertex.index, vertex.index]
        
        ppr_scores[ [[v] for v in cluster] , cluster] = cc_ppr_scores

    return ppr_scores

