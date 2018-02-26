""" mainly clustering class """
import copy
from collections import defaultdict
import networkx as nx

from clusim.dag import DAG, Dendrogram

class Clustering(object):
    """Base class for clusterings.

    Parameters
    ----------
    elm2clus_dict : dict, optional
        Initialize based on an elm2clus_dict: { elementid: [clu1, clu2, ... ] }.  
        Each element is a key with value a list of clusters to which it belongs.

    clus2elm_dict : dict, optional
        Initialize based on an clus2elm_dict: { clusid: [el1, el2, ... ]}.  
        Each cluster is a key with value a list of elements which belong to it.
    """

    def __init__(self, elm2clus_dict=None, clus2elm_dict=None):

        self.empty_start()
        
        if elm2clus_dict is not None:
            # create clustering from elm2clus_dict
            self.from_elm2clus_dict(elm2clus_dict)

        elif clus2elm_dict is not None:
            # create clustering from clus2elm_dict
            self.from_clus2elm_dict(clus2elm_dict)

    def empty_start(self):
        # number of elements
        self.n_elements = 0
        # number of clusters
        self.n_clusters = 0

        # list of elements
        self.elements = []
        # list of clusters
        self.clusters = []

        # representation as an elm2clus_dict
        self.elm2clus_dict = defaultdict(set)
        # representation as an clus2elm_dict
        self.clus2elm_dict = defaultdict(set)

        # cluster size sequence
        self.clus_size_seq = []
        # disjoint partitions?
        self.is_disjoint = False
        # hierarchical partitions?
        self.is_hierarchical = False


    def from_elm2clus_dict(self, elm2clus_dict):
        """
        This method creates a clustering from an elm2clus_dict dictionary:
        { elementid: [clu1, clu2, ... ] } where each element is a key with 
        value a list of clusters to which it belongs.  Clustering features
        are then calculated.

        Parameters
        ----------
        elm2clus_dict : dict
            { elementid: [clu1, clu2, ... ] }

        >>> import clusim
        >>> elm2clus_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]}
        >>> clu = Clustering()
        >>> clu.from_elm2clus_dict(elm2clus_dict)
        >>> print_clustering(clu)
        """

        self.elm2clus_dict = copy.deepcopy(elm2clus_dict)
        self.elements = list(self.elm2clus_dict.keys())
        self.n_elements = len(self.elements)

        self.clus2elm_dict = self.to_clus2elm_dict()
        self.clusters = list(self.clus2elm_dict.keys())
        self.n_clusters = len(self.clusters)

        self.clus_size_seq = self.find_clus_size_seq()

        self.is_disjoint = self.find_num_overlap() == 0

    def from_clus2elm_dict(self, clus2elm_dict):
        """
        This method creates a clustering from an clus2elm_dict dictionary:
        { clusid: [el1, el22, ... ] } where each cluster is a key with 
        value a list of elements which belong to it.  Clustering features
        are then calculated.

        Parameters
        ----------
        clus2elm_dict : dict
            { clusid: [el1, el2, ... ] }


        >>> import clusim
        >>> clus2elm_dict = {0:[0,1,2], 1:[2,3], 2:[4,5]}
        >>> clu = Clustering()
        >>> clu.from_clus2elm_dict(clus2elm_dict)
        >>> print_clustering(clu)
        """

        self.clus2elm_dict = copy.deepcopy(clus2elm_dict)
        self.clusters =  list(self.clus2elm_dict.keys())
        self.n_clusters = len(self.clusters)

        self.elm2clus_dict = self.to_elm2clus_dict()
        self.elements = list(self.elm2clus_dict.keys())
        self.n_elements = len(self.elements)

        self.clus_size_seq = self.find_clus_size_seq()

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

        >>> import clusim
        >>> cluster_list = [ [0,1,2], [2,3], [4,5]]
        >>> clu = Clustering()
        >>> clu.from_cluster_list(cluster_list)
        >>> print_clustering(clu)
        """
        self.from_clus2elm_dict({iclus:set(clist) for iclus, clist in enumerate(cluster_list)})

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
        return list(map(list, self.clus2elm_dict.values()))
        
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

        >>> import clusim
        >>> membership_list = [0,0,0,1,2,2]
        >>> clu = Clustering()
        >>> clu.from_membership_list(membership_list)
        >>> print_clustering(clu)
        """ 
        self.from_elm2clus_dict({elm:set([clu]) for elm, clu in enumerate(membership_list)})

    def clustering_from_igraph_cover(self, igraphcover):
        """
        This method creates a clustering from an igraph VertexCover object.
        See the :class:`igraph.Cover.VertexCover` class.
        Clustering features are then calculated.

        """ 
        igc = igraphcover.as_cover().membership
        self.from_elm2clus_dict({elm:set(clu) for elm, clu in enumerate(igc)})

    def to_clus2elm_dict(self):
        """ 
        Create a clu2elm_dict: {clusterid: [el1, el2, ... ]} from the
        stored elm2clus_dict.
        """

        clus2elm_dict = defaultdict(set)
        for elm in self.elm2clus_dict:
            for clu in self.elm2clus_dict[elm]:
                clus2elm_dict[clu].add(elm)

        return clus2elm_dict

    def to_elm2clus_dict(self):
        """ 
        Create a elm2clus_dict: {elementid: [clu1, clu2, ... ]} from the
        stored clu2elm_dict.
        """

        elm2clus_dict = defaultdict(set)
        for clu in self.clus2elm_dict:
            for elm in self.clus2elm_dict[clu]:
                elm2clus_dict[elm].add(clu)

        return elm2clus_dict

    def find_clus_size_seq(self):
        """ 
        This method finds the cluster size sequence for the clustering. 

        Returns
        -------
        clus_size_seq : list of integers
            A list where the ith entry corresponds to the size of the ith
            cluster.
        
        >>> import clusim
        >>> elm2clus_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]}
        >>> clu = Clustering(elm2clus_dict = elm2clus_dict)
        >>> print("Cluster Size Sequence:", clu.find_clus_size_seq())
        * Cluster Size Sequence: [3, 2, 2]
        """
        return [len(self.clus2elm_dict[clu]) for clu in self.clusters]

    def find_num_overlap(self):
        """ 
        This method finds the number of elements which are in more than one 
        cluster in the clustering. 

        Returns
        -------
        clus_size_seq : integer
            The number of elements in at least two clusters.
        
        >>> import clusim
        >>> elm2clus_dict = {0:[0], 1:[0], 2:[0,1], 3:[1], 4:[2], 5:[2]}
        >>> clu = Clustering(elm2clus_dict = elm2clus_dict)
        >>> print("Overlap size:", clu.find_num_overlap())
        * Overlap size: 1
        """
        return sum([len(self.elm2clus_dict[elm]) > 1 for elm in self.elements])

    def merge_clusters(self, c1, c2, new_name = None):
        if new_name is None:
            new_name = c1

        new_clus = self.clus2elm_dict[c1] | self.clus2elm_dict[c2]
        del self.clus2elm_dict[c1]
        del self.clus2elm_dict[c2]

        self.clus2elm_dict[new_name] = new_clus
        self.from_clus2elm_dict(self.clus2elm_dict)
        

class HierClustering(Clustering):

    def __init__(self, elm2clus_dict=None, clus2elm_dict=None, hier_graph = None):
        
        self.empty_start()
        self.is_hierarchical = True
        self.hier_clusdict = None
        
        if not elm2clus_dict is None:
            self.from_elm2clus_dict(elm2clus_dict)
            
        elif not clus2elm_dict is None:
            self.from_clus2elm_dict(clus2elm_dict)
            
        if not (hier_graph is None):
            self.from_digraph(hier_graph)

        
    def from_digraph(self, hier_graph = None):
        if True: #nx.is_acyclic(hier_graph):
            self.hiergraph = hier_graph
            self.clusters = list(hier_graph.nodes())
            self.n_clusters = len(self.clusters)
        else:
            print("Graph must be acyclic!")
            
    def from_linkage(self, linkage_matrix, dist_rescaled = False):
        self.hiergraph = Dendrogram().from_linkage(linkage_matrix, dist_rescaled)
        
    def cut_at_depth(self, depth = 0, cuttype = 'shortestpath', rescale_path_type = 'max'):
        clusters_at_depth = self.hiergraph.cut_at_depth(depth = depth, 
                                                        cuttype = cuttype, 
                                                        rescale_path_type = rescale_path_type)
        
        new_cluster_dict = {c:self.downstream_elements(c) for c in clusters_at_depth}
        flat_clustering = Clustering(clus2elm_dict = new_cluster_dict)
        return flat_clustering
    
    def downstream_elements(self, cluster):
        try:
            return self.clus2elm_dict[cluster]
        except KeyError:
            el = set([])
            for c in nx.dfs_preorder_nodes(self.hiergraph, cluster):
                try:
                    el.update(self.clus2elm_dict[c])
                except KeyError:
                    pass

            return el
    
    def hierclusdict(self):
        if self.hier_clusdict is None:
            self.hier_clusdict = {}
            for cluster in self.hiergraph.nodes():
                self.hier_clusdict[cluster] = self.downstream_elements(cluster)
        return self.hier_clusdict
    
