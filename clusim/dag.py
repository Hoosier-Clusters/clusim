import networkx as nx
import numpy as np
import matplotlib.pylab as plt

try:
    import dendropy
except ImportError:
    print("DendroPY not supported.")

class DAG(nx.DiGraph):
    
    def roots(self):
        return [node for node in self.nodes() if self.in_degree(node) == 0]

    def leaves(self):
        return [node for node in self.nodes() if self.out_degree(node) == 0]
    
    def maxdist_from_roots(self, graph = None):
        if graph is None:
            graph = self
            
        dist = {} # stores [node, distance] pair
        for node in nx.topological_sort(graph):
            # pairs of dist,node for all incoming edges
            pairs = [(dist[v][0]+1,v) for v in graph.pred[node]] 
            if pairs:
                dist[node] = max(pairs)
            else:
                dist[node] = (0, node)

        return dist
    
    def mindist_from_roots(self, graph = None):
        if graph is None:
            graph = self
            
        dist = {} # stores [node, distance] pair
        for node in nx.topological_sort(graph):
            # pairs of dist,node for all incoming edges
            pairs = [(dist[v][0]+1,v) for v in graph.pred[node]] 
            if pairs:
                dist[node] = min(pairs)
            else:
                dist[node] = (0, node)

        return dist
    
    def height(self):
        dfr = self.maxdist_from_roots()
        max_dist  = max([disttuple[0] for disttuple in dfr.itervalues()]) + 1
        return max_dist
    
    def cut_at_depth(self, depth = None, cuttype = 'shortestpath', rescale_path_type = 'max'):
        if cuttype == 'shortestpath':
            dfr = self.mindist_from_roots()
            node_dist = {node:dfr[node][0] for node in self.nodes()}
        elif cuttype == 'rescaled':
            node_dist = self.rescale(rescale_path_type)

        cluster_list = [node for node in self.nodes() if node_dist[node] == depth]

        return cluster_list
    
    def rescale(self, rescale_path_type = 'max'):

        if rescale_path_type == 'max':
            dfr = self.maxdist_from_roots(self)
            dtl = self.maxdist_from_roots(self.reverse(copy = True))
        elif rescale_path_type == 'min':
            dfr = self.mindist_from_roots(self)
            dtl = self.mindist_from_roots(self.reverse(copy = True))
        elif rescale_path_type == 'linkage':
            try:
                return self.linkage_dist
                print("using linkage")
            except:
                return collections.defaultdict(int)

        rescaled_level = {}
        for node in self.nodes():
            path_to_node = dfr[node][0]
            path_from_node = dtl[node][0]
            total_path_len = path_to_node + path_from_node
            if total_path_len == 0.0:
                rescaled_level[node] = 0.0
            else:
                rescaled_level[node] = float(path_to_node) / total_path_len

        return rescaled_level
    
    def draw(self, rescaled_level = None, rescale_path_type = 'max', ax = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        treepos= nx.drawing.nx_agraph.graphviz_layout(self, prog='dot')
        treeys = [pt[1] for pt in treepos.values()]
        mintreey = min(treeys)
        maxtreey = max(treeys)

        if rescaled_level == True:
            rescaled_level = self.rescale(rescale_path_type = rescale_path_type)
            treepos = {node:(pt[0], mintreey + (1.0 - rescaled_level[node])*(maxtreey- mintreey)) for node, pt in treepos.items()}
        elif type(rescaled_level) is dict:
            treepos = {node:(pt[0], mintreey + (1.0 - rescaled_level[node])*(maxtreey- mintreey)) for node, pt in treepos.items()}

        nx.draw(self, pos = treepos, arrows = False, ax = ax, **kwargs)
        
        return ax
    
    def _tree_edges(self, n,r):
        # From http://stackoverflow.com/questions/26896370/faster-way-to-build-a-tree-graph-using-networkx-python
        # helper function for trees
        # yields edges in rooted tree at 0 with n nodes and branching ratio r
        nodes=iter(range(n))
        parents=[next(nodes)] # stack of max length r
        while parents:
            source=parents.pop(0)
            for i in range(r):
                try:
                    target=next(nodes)
                    parents.append(target)
                    yield source,target
                except StopIteration:
                    break

    def make_regular_tree(self, N = 1, r = 2):
        self.add_edges_from(list(self._tree_edges(N, r)))
        self.linkage_dist = {n:d for n, (d,_) in self.maxdist_from_roots().items()}

    def make_complete_rary_tree(self, h = 2, r = 2):
        self.add_edges_from(nx.balanced_tree(h = h, r = r, create_using = nx.DiGraph()).edges)
        self.linkage_dist = {n:d for n, (d,_) in self.maxdist_from_roots().items()}

    def swap_nodes(self, n1, n2, swap_parents = True, swap_children = False):
        if swap_parents:
            n1parents = list(self.predecessors(n1))
            n2parents = list(self.predecessors(n2))
            self.add_edges_from([(p, n1) for p in n2parents])
            self.remove_edges_from([(p, n1) for p in n1parents])

            self.add_edges_from([(p, n2) for p in n1parents])
            self.remove_edges_from([(p, n2) for p in n2parents])
            

class Dendrogram(DAG):

    def from_linkage(self, linkage_matrix, dist_rescaled = False):
        N = linkage_matrix.shape[0] + 1

        if dist_rescaled:
            maxdist = max(linkage_matrix[:,2])
            distances = 1.0 - linkage_matrix[:, 2]/maxdist
            linkage_dist = {ipt:1.0 for ipt in range(N)}
        else:
            distances = linkage_matrix[:, 2]
            linkage_dist = {ipt:0.0 for ipt in range(N)}

        for iclus in range(N - 1):
            clus_id = N + iclus 
            linkage_dist[clus_id] = distances[iclus]
            self.add_edges_from([(clus_id, int(linkage_matrix[iclus,0])), (clus_id, int(linkage_matrix[iclus,1]))])

        self.linkage_dist = linkage_dist

    def to_dendropy_tree(self, taxon_namespace, weighted = False):
        tree = dendropy.Tree(taxon_namespace=taxon_namespace)

        seed_node = self.roots()[0]
        
        if weighted:
            edge_length = lambda par, child: np.abs(self.linkage_dist[par] - self.linkage_dist[child])
        else:
            edge_length = lambda par, child: 1.0
        
        tree_dict = {seed_node:tree.seed_node}
        for clus in nx.topological_sort(self):
            for child in self.successors(clus):
                tree_dict[child] = tree_dict[clus].new_child(edge_length = edge_length(clus, child))

        for clus in self.leaves():
            tree_dict[clus].taxon = taxon_namespace.get_taxon(str(clus))
        
        return tree

    def from_dendropy_tree(self, tree, keep_taxon = True):

        linkage_dist = {}
        for i, node in enumerate(tree.levelorder_node_iter()):
            
            if keep_taxon and node.taxon is not None:
                node_name = node.taxon.label
            else:
                node_name = i
            node.label = node_name
                
            if node.parent_node is not None:
                self.add_edge(node.parent_node.label, node.label, weight = node.edge_length)
                linkage_dist[node.label] = linkage_dist[node.parent_node.label] + node.edge_length
            else:
                linkage_dist[node.label] = 0.0

        self.linkage_dist = linkage_dist

    def from_newick(self, s, taxon_namespace, keep_taxon = True):
        tree = dendropy.Tree(taxon_namespace=taxon_namespace)
        tree = tree.get(data=s, schema="newick")
        self.from_dendropy_tree(tree, keep_taxon = keep_taxon)


    def make_random_dendrogram(self, N = 10):
        self.make_biased_dendrogram(N = N, p = 0.0)


    def make_biased_dendrogram(self, N = 10, p = 1.0):

        self.add_node(0)
        leaves = self.leaves()
        i = 1
        while len(leaves) < N:
            if np.random.random() > p:
                parent = leaves.pop(np.random.randint(0,len(leaves)))
            else:
                parent = leaves.pop(-1)

            self.add_edges_from([(parent, i), (parent, i + 1)])
            leaves.extend([i, i + 1])
            i += 2






