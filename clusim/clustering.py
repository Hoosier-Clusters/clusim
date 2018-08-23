""" mainly clustering class """
import copy
from collections import defaultdict

import numpy as np
import networkx as nx

from clusim.dag import Dendrogram


class ClusterError(ValueError):
    """Exception raised for errors in the clustering.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Clustering(object):
    """Base class for clusterings.

    Parameters
    ----------
    elm2clu_dict : dict, optional
        Initialize based on an elm2clu_dict: { elementid: [clu1, clu2, ... ] }.
        The value is a list of clusters to which the element belongs.

    clu2elm_dict : dict, optional
        Initialize based on an clu2elm_dict: { clusid: [el1, el2, ... ]}.
        Each cluster is a key with value a list of elements which belong to it.

    hier_graph : networkx graph, optional
        Initialize based on a hierarchical acyclic graph capturing the cluster
        membership at each scale.
    """

    def __init__(self, elm2clu_dict=None, clu2elm_dict=None, hier_graph=None):

        self.empty_start()

        if elm2clu_dict is not None:
            # create clustering from elm2clu_dict
            self.from_elm2clu_dict(elm2clu_dict)

        elif clu2elm_dict is not None:
            # create clustering from clu2elm_dict
            self.from_clu2elm_dict(clu2elm_dict)

        if hier_graph is not None:
            self.is_hierarchical = True
            self.hier_graph = hier_graph
            self.hier_clusdict()

    def empty_start(self):
        # number of elements
        self.n_elements = 0
        # number of clusters
        self.n_clusters = 0

        # list of elements
        self.elements = []
        # list of clusters
        self.clusters = []

        # representation as an elm2clu_dict
        self.elm2clu_dict = defaultdict(set)
        # representation as an clu2elm_dict
        self.clu2elm_dict = defaultdict(set)
        # represetation as an acyclic graph
        self.hier_graph = nx.DiGraph()

        # cluster size sequence
        self.clu_size_seq = []
        # disjoint partitions?
        self.is_disjoint = False
        # hierarchical partitions?
        self.is_hierarchical = False
        # representation as an hiercluster dict
        self.hierclusdict = None

    def copy(self):
        """
        Return a copy of the clustering.

        >>> import from clusim.clustering import Clustering
        >>> from clusim.plotutils import print_clustering
        >>> clu = clusim.Clustering()
        >>> clu2 = clu.copy()
        >>> print_clustering(clu)
        >>> print_clustering(clu)
        """
        return copy.deepcopy(self)

    def from_elm2clu_dict(self, elm2clu_dict):
        """
        This method creates a clustering from an elm2clu_dict dictionary:
        { elementid: [clu1, clu2, ... ] } where each element is a key with
        value a list of clusters to which it belongs.  Clustering features
        are then calculated.

        Parameters
        ----------
        elm2clu_dict : dict
            { elementid: [clu1, clu2, ... ] }

        >>> from clusim.clustering import Clustering
        >>> from clusim.plotutils import print_clustering
        >>> elm2clu_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]}
        >>> clu = Clustering()
        >>> clu.from_elm2clu_dict(elm2clu_dict)
        >>> print_clustering(clu)
        """

        self.elm2clu_dict = copy.deepcopy(elm2clu_dict)
        self.elements = list(self.elm2clu_dict.keys())
        self.n_elements = len(self.elements)

        self.clu2elm_dict = self.to_clu2elm_dict()
        self.clusters = list(self.clu2elm_dict.keys())
        self.n_clusters = len(self.clusters)

        self.clu_size_seq = self.find_clu_size_seq()

        self.is_disjoint = self.find_num_overlap() == 0

    def from_clu2elm_dict(self, clu2elm_dict):
        """
        This method creates a clustering from an clu2elm_dict dictionary:
        { clusid: [el1, el22, ... ] } where each cluster is a key with
        value a list of elements which belong to it.  Clustering features
        are then calculated.

        Parameters
        ----------
        clu2elm_dict : dict
            { clusid: [el1, el2, ... ] }


        >>> import from clusim.clustering import Clustering
        >>> from clusim.plotutils import print_clustering
        >>> clu2elm_dict = {0:[0,1,2], 1:[2,3], 2:[4,5]}
        >>> clu = Clustering()
        >>> clu.from_clu2elm_dict(clu2elm_dict)
        >>> print_clustering(clu)
        """

        self.clu2elm_dict = copy.deepcopy(clu2elm_dict)
        self.clusters = list(self.clu2elm_dict.keys())
        self.n_clusters = len(self.clusters)

        self.elm2clu_dict = self.to_elm2clu_dict()
        self.elements = list(self.elm2clu_dict.keys())
        self.n_elements = len(self.elements)

        self.clu_size_seq = self.find_clu_size_seq()

        self.is_disjoint = self.find_num_overlap() == 0

    def from_cluster_list(self, cluster_list):
        """
        This method creates a clustering from a cluster list:
        [ [el1, el2, ...], [el5, ...], ... ],  a list of lists,
        where each inner list corresponds to the elements in
        a cluster.  Clustering features are then calculated.

        Parameters
        ----------
        cluster_list : list of lists
            [ [el1, el2, ...], [el5, ...], ... ]

        >>> import from clusim.clustering import Clustering
        >>> from clusim.plotutils import print_clustering
        >>> cluster_list = [ [0,1,2], [2,3], [4,5]]
        >>> clu = Clustering()
        >>> clu.from_cluster_list(cluster_list)
        >>> print_clustering(clu)
        """
        self.from_clu2elm_dict({iclus: set(clist)
                               for iclus, clist in enumerate(cluster_list)})

    def to_cluster_list(self):
        """
        This method returns a clustering in cluster list format:
        [ [el1, el2, ...], [el5, ...], ... ],  a list of lists,
        where each inner list corresponds to the elements in
        a cluster.

        Returns
        ----------
        cluster_list : list of lists
            [ [el1, el2, ...], [el5, ...], ... ]

        """
        return list(map(list, self.clu2elm_dict.values()))

    def from_membership_list(self, membership_list):
        """
        This method creates a clustering from a membership list:
        [ clu_for_el1, clu_for_el2, ... ],  a list of integers
        the ith entry corresponds to the cluster membership of the ith element.
        Clustering features are then calculated.

        .. note:: Membership Lists can only represent partitions (no overlaps)

        Parameters
        ----------
        cluster_list : list of integers
             clu_for_el1, clu_for_el2, ... ]

        >>> import from clusim.clustering import Clustering
        >>> from clusim.plotutils import print_clustering
        >>> membership_list = [0,0,0,1,2,2]
        >>> clu = Clustering()
        >>> clu.from_membership_list(membership_list)
        >>> print_clustering(clu)
        """
        self.from_elm2clu_dict({elm: set([clu])
                                for elm, clu in enumerate(membership_list)})

    def to_membership_list(self):
        """
        This method returns the clustering as a membership list:
        [ clu_for_el1, clu_for_el2, ... ],  a list of integers
        the ith entry corresponds to the cluster membership of the ith element.

        .. note:: Membership Lists can only represent partitions (no overlaps)
        """

        if not self.is_disjoint:
            raise ClusterError("", "Membership Lists can only be created for "
                                   "disjoint clusterings. Your clustering "
                                   "contains overlaps.")

        elif self.is_hierarchical:
            raise ClusterError("", "Membership Lists can only be created for "
                                   "disjoint clusterings. Your clustering is "
                                   "hierarchical.")

        else:
            return [list(self.elm2clu_dict[elm])[0]
                    for elm in sorted(self.elements)]

    def clustering_from_igraph_cover(self, igraphcover):
        """
        This method creates a clustering from an igraph VertexCover object.
        See the :class:`igraph.Cover.VertexCover` class.
        Clustering features are then calculated.

        """
        igc = igraphcover.as_cover().membership
        self.from_elm2clu_dict({elm: set(clu) for elm, clu in enumerate(igc)})
        return self

    def to_clu2elm_dict(self):
        """
        Create a clu2elm_dict: {clusterid: [el1, el2, ... ]} from the
        stored elm2clu_dict.
        """

        clu2elm_dict = defaultdict(set)
        for elm in self.elm2clu_dict:
            for clu in self.elm2clu_dict[elm]:
                clu2elm_dict[clu].add(elm)

        return clu2elm_dict

    def to_elm2clu_dict(self):
        """
        Create a elm2clu_dict: {elementid: [clu1, clu2, ... ]} from the
        stored clu2elm_dict.
        """

        elm2clu_dict = defaultdict(set)
        for clu in self.clu2elm_dict:
            for elm in self.clu2elm_dict[clu]:
                elm2clu_dict[elm].add(clu)

        return elm2clu_dict

    def find_clu_size_seq(self):
        """
        This method finds the cluster size sequence for the clustering.

        Returns
        -------
        clu_size_seq : list of integers
            A list where the ith entry corresponds to the size of the ith
            cluster.

        >>> import from clusim.clustering import Clustering
        >>> elm2clu_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]}
        >>> clu = Clustering(elm2clu_dict = elm2clu_dict)
        >>> print("Cluster Size Sequence:", clu.find_clu_size_seq())
        * Cluster Size Sequence: [3, 2, 2]
        """
        return [len(self.clu2elm_dict[clu]) for clu in self.clusters]

    def find_num_overlap(self):
        """
        This method finds the number of elements which are in more than one
        cluster in the clustering.

        Returns
        -------
        clu_size_seq : integer
            The number of elements in at least two clusters.

        >>> import from clusim.clustering import Clustering
        >>> elm2clu_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]}
        >>> clu = Clustering(elm2clu_dict = elm2clu_dict)
        >>> print("Overlap size:", clu.find_num_overlap())
        * Overlap size: 1
        """
        return sum([len(self.elm2clu_dict[elm]) > 1 for elm in self.elements])

    def merge_clusters(self, c1, c2, new_name=None):
        """
        This method merges the elements in two clusters from the clustering.
        The merged clustering will be named new_name if provided, otherwise
        it will assume the name of cluster c1.

        Returns
        -------
        self : Clustering

        >>> import from clusim.clustering import Clustering
        >>> from clusim.plotutils import print_clustering
        >>> elm2clu_dict = {0:[0], 1:[0], 2:[0], 3:[1], 4:[2], 5:[2]}
        >>> clu = Clustering(elm2clu_dict = elm2clu_dict)
        >>> print_clustering(clu)
        >>> clu.merge_clusters(1,2, new_name = 3)
        >>> print_clustering(clu)
        """
        if new_name is None:
            new_name = c1

        new_clus = self.clu2elm_dict[c1] | self.clu2elm_dict[c2]
        del self.clu2elm_dict[c1]
        del self.clu2elm_dict[c2]

        self.clu2elm_dict[new_name] = new_clus
        self.from_clu2elm_dict(self.clu2elm_dict)
        return self

    ##############################################
    # extra support for hierarchical clusterings
    ##############################################

    def downstream_elements(self, cluster):
        """
        This method finds all elements contained in a cluster from a
        hierarchical clustering by visiting all downstream clusters
        and adding their elements.

        Parameters
        ----------
        cluster : the name of the parent cluster

        Returns
        -------
        el : element list


        """
        if cluster in self.hier_graph.leaves():
            return self.clu2elm_dict[cluster]
        else:
            el = set([])
            for c in nx.dfs_preorder_nodes(self.hier_graph, cluster):
                try:
                    el.update(self.clu2elm_dict[c])
                except KeyError:
                    pass

            return el

    def from_scipy_linkage(self, linkage_matrix, dist_rescaled=False):
        """
        This method creates a clustering from a scipy linkage object resulting
        from the agglomerative hierarchical clustering.
        Clustering features are then calculated.

        Parameters
        ----------
        linkage_matrix : numpy matrix
            the linkage matrix from scipy

        dist_rescaled : Boolean
            if True, the linkage distances are linearlly rescaled to be
            in-between 0 and 1


        >>> import from clusim.clustering import Clustering
        >>> from scipy.cluster.hierarchy import dendrogram, linkage
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data1 = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],
                                                  size=[100,])
        >>> data2 = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]],
                                                  size=[50,])
        >>> Xdata = np.concatenate((data1, data2), )
        >>> Z = linkage(X, 'ward')
        >>> clu = Clustering()
        >>> clu.from_scipy_linkage(Z, dist_rescaled=False)
        """
        self.hier_graph = Dendrogram().from_linkage(linkage_matrix,
                                                    dist_rescaled)

    def to_dendropy_tree(self, taxon_namespace, weighted=False):
        import dendropy

        tree = dendropy.Tree(taxon_namespace=taxon_namespace)

        seed_node = self.hier_graph.roots()[0]

        if weighted:
            def edge_length(par, child):
                return np.abs(self.hier_graph.linkage_dist[par] -
                              self.hier_graph.linkage_dist[child])
        else:
            def edge_length(par, child):
                return 1.0

        tree_dict = {seed_node: tree.seed_node}
        for clus in nx.topological_sort(self.hier_graph):
            for child in self.hier_graph.successors(clus):
                tree_dict[child] = tree_dict[clus].new_child(
                    edge_length=edge_length(clus, child))

        for clus in self.hier_graph.leaves():
            tree_dict[clus].taxon = taxon_namespace.get_taxon(
                str(list(self.clu2elm_dict[clus])[0]))

        return tree

    def from_digraph(self, hier_graph=None):
        if True:  # nx.is_acyclic(hier_graph):
            self.hier_graph = hier_graph
            self.clusters = list(hier_graph.nodes())
            self.n_clusters = len(self.clusters)
        else:
            print("Graph must be acyclic!")

    def cut_at_depth(self, depth=0, cuttype='shortestpath',
                     rescale_path_type='max'):
        clusters_at_depth = self.hier_graph.cut_at_depth(
            depth=depth, cuttype=cuttype, rescale_path_type=rescale_path_type)

        new_cluster_dict = {c: self.downstream_elements(c)
                            for c in clusters_at_depth}
        flat_clustering = Clustering(clu2elm_dict=new_cluster_dict)
        return flat_clustering

    def hier_clusdict(self):
        if self.hierclusdict is None:
            self.hierclusdict = {}
            for cluster in self.hier_graph.nodes():
                self.hierclusdict[cluster] = self.downstream_elements(cluster)
            self.clusters = list(self.hierclusdict.keys())
            self.n_clusters = len(self.clusters)
        return self.hierclusdict
