""" clustering generators """
from collections import defaultdict
import numpy as np
import itertools
import mpmath
import copy

from clustering import Clustering


def make_equal_clustering(n_elements, n_clusters):
    """
        This function creates a random clustering with equally sized clusters.
        If n_elements % n_clusters != 0, cluster sizes will differ by one element.

        Parameters
        ----------
        n_elements : int
            The number of elements

        n_clusters : int
            The number of clusters

        Returns
        -------
        new_clsutering : Clustering
            The new clustering with equally sized clusters.

        >>> import clusim
        >>> clu = make_equal_clustering(n_elements = 9, n_clusters = 3)
        >>> print_clustering(clu)
    """
    new_elm2clus_dict = {el:[el%n_clusters] for el in range(n_elements)}
    new_clustering = Clustering(new_elm2clus_dict)
    return new_clustering

def make_random_clustering(n_elements, n_clusters=1, random_model = 'all', tol = 1.0e-15):
    """
        This function creates a random clustering according to one of three random models.

        Parameters
        ----------
        n_elements : int
            The number of elements

        n_clusters : int
            The number of clusters

        random_model : string
            The random model to use:

            'all' : uniform distrubtion over the set of all clusterings of n_elements

            'num' : uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'uniform' : uniform distrubtion over the set of all clusterings with n_elements

        tol : float, optional
            The tolerance used by the algorithm for 'all' clusterings

        Returns
        -------
        new_clsutering : Clustering
            The new clustering.

        >>> import clusim
        >>> clu = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print_clustering(clu)
    """
    if random_model == 'all':
        new_clustering = generate_random_partition_all(n_elements = n_elements, tol = tol)
    
    elif random_model == 'num':
        new_cluster_list = generate_random_partition_num(n_elements = n_elements, n_clusters = n_clusters)
        new_clustering = Clustering()
        new_clustering.from_cluster_list(new_cluster_list)

    elif random_model == 'uniform':
        clu_list = range(n_clusters)
        new_elm2clus_dict = {el:[np.random.choice(clu_list)] for el in range(n_elements)}
        new_clustering = Clustering(new_elm2clus_dict)
    
    return new_clustering

def make_singleton_clustering(n_elements):
    """
        This function creates a clustering with each element in its own cluster.

        Parameters
        ----------
        n_elements : int
            The number of elements

        Returns
        -------
        new_clsutering : Clustering
            The new clustering.

        >>> import clusim
        >>> clu = make_singleton_clustering(n_elements = 9)
        >>> print_clustering(clu)
    """
    new_clsutering = make_regular_clustering(n_elements = n_elements, n_clusters = n_elements)
    return new_clsutering

def shuffle_memberships(clustering, percent = 1.0):
    """
        This function creates a new clustering by shuffling the element memberships from the original clustering.

        Parameters
        ----------
        clustering : Clustering
            The original clustering.

        percent : float, optional (default 1.0)
            The fractional percentage (between 0.0 and 1.0) of the elements to shuffle.

        Returns
        -------
        new_clsutering : Clustering
            The new clustering.

        >>> import clusim
        >>> orig_clu = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print_clustering(orig_clu)
        >>> shuffle_clu = shuffle_memberships(orig_clu, percent = 0.5)
        >>> print_clustering(shuffle_clu)
    """
    el_to_shuffle = np.random.choice(clustering.elements, int(percent * clustering.n_elements), replace=False)
    shuffled_el = np.random.permutation(el_to_shuffle)
    newkeys = dict(zip(el_to_shuffle, shuffled_el))

    new_elm2clus_dict = defaultdict(set)
    for el in clustering.elements:
        if el in shuffled_el:
            new_elm2clus_dict[newkeys[el]] = clustering.elm2clus_dict[el]
        else:
            new_elm2clus_dict[el] = clustering.elm2clus_dict[el]

    new_clustering = Clustering(new_elm2clus_dict)
    return new_clustering

def shuffle_memberships_pa(clustering, Nsteps = 1, constant_num_clusters = True):
    """
        This function creates a new clustering by shuffling the element memberships 
        from the original clustering according to the preferential attachment model.

        Parameters
        ----------
        clustering : Clustering
            The original clustering.

        Nsteps : int, optional (default 1)
            The number of times to run the preferential attachment algorithm.

        constant_num_clusters : boolean, optional (default True)
            Reject a shuffling move if it leaves a cluster with no elements. This will
            keep the number of clusters constant.

        Returns
        -------
        new_clsutering : Clustering
            The new clustering.

        >>> import clusim
        >>> orig_clu = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print_clustering(orig_clu)
        >>> shuffle_clu = shuffle_memberships_pa(orig_clu, Nsteps = 10, constant_num_clusters = True)
        >>> print_clustering(shuffle_clu)
    """
    n_elements_norm  = 1./float(clustering.n_elements)

    Nclusters = clustering.n_clusters

    cluster_list = clustering.to_cluster_list()
    cluster_size_prob = np.array(map(len, cluster_list)) * n_elements_norm
    clusternames = range(Nclusters)

    for istep in range(Nsteps):
        
        from_cluster = np.random.choice(clusternames, p = cluster_size_prob)
        if cluster_size_prob[from_cluster] > 1.5*n_elements_norm or not constant_num_clusters:

            exchanged_element = np.random.choice(cluster_list[from_cluster], 1, replace=False)[0]
            new_cluster = np.random.choice(clusternames, p = cluster_size_prob)

            if new_cluster != from_cluster:
                cluster_list[from_cluster].remove(exchanged_element)
                cluster_size_prob[from_cluster] -= n_elements_norm

                cluster_list[new_cluster].append(exchanged_element)
                cluster_size_prob[new_cluster] += n_elements_norm

    new_clustering = Clustering()
    new_clustering.clustering_from_cluster_list(cluster_list)

    return new_clustering

def generate_random_partition_num(n_elements, n_clusters):
    """
        A recursive function to generate a random partition uniformly selected from 'Num',
        the set of all clusterings with n_elements in n_clusters.

        Based on this blog post: `b link`_.

        .. _b link: http://thousandfold.net/cz/2013/09/25/sampling-uniformly-from-the-set-of-partitions-in-a-fixed-number-of-nonempty-sets/

        Parameters
        ----------
        n_elements : int
            The number of elements

        n_clusters : int
            The number of clusters

        Returns
        -------
        new_clsutering : Clustering
            The new clustering.

        >>> import clusim
        >>> clu = generate_random_partition_num(n_elements = 10, n_clusters = 3)
        >>> print_clustering(clu)
    """

    assert n_clusters <= n_elements

    if n_elements == 1:
        current_partition = [[0]]

    else:
        stirling_prob = mpmath.stirling2(n_elements - 1, n_clusters - 1) / mpmath.stirling2(n_elements, n_clusters)

        if np.random.random() < stirling_prob:
            current_partition = generate_random_partition_num(n_elements = n_elements - 1, n_clusters = n_clusters - 1)
            current_partition.append([n_elements - 1])
        else:
            current_partition = generate_random_partition_num(n_elements = n_elements - 1, n_clusters = n_clusters)
            current_clu = np.random.randint(n_clusters)
            current_partition[current_clu].append(n_elements - 1)

    return current_partition


all_partition_weight_dict = {}
def generate_random_partition_all(n_elements, tol = 1.0e-15):
    """
        This function creates a random clustering according to the 'All' random model
        by uniformly selecting a clustering from the set of all clusterings with n_elements.

        Parameters
        ----------
        n_elements : int
            The number of elements

        tol : float, optional
            The tolerance used by the algorithm for 'all' clusterings

        Returns
        -------
        new_clsutering : Clustering
            The new clustering.

        >>> import clusim
        >>> clu = generate_random_partition_all(n_elements = 9)
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

    K = np.random.choice(np.arange(1, len(weights) + 1), p = weights)
    colors = np.random.randint(K, size = n_elements)

    new_clustering = Clustering()
    new_clustering.from_membership_list(colors)
    return new_clustering

def clustering_ensemble_generator(clustering, random_model = 'perm'):
    """
        A generator for every partition in one of three random models.

        Parameters
        ----------
        clustering : Clustering
            The seed clustering

        random_model : string
            The random model to use:

            'all' : uniform distrubtion over the set of all clusterings of n_elements

            'num' : uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'perm' : uniform distrubtion over all permutations of the elements within the clustering

        Returns
        -------
        new_clustering: Clustering
            The new clustering.

        >>> import clusim
        >>> seed_clu = make_equal_clustering(n_elements = 9, n_clusters = 3)
        >>> for clu in clustering_ensemble_generator(seed_clu, random_model = 'perm'):
        >>>     print_clustering(clu)
    """

    if random_model == 'all':
        num_clusters_range = range(1, clustering.n_elements + 1)
    elif random_model == 'perm' or random_model == 'num':
        num_clusters_range = [clustering.n_clusters]
    else:
        print("Random model not supported")

    sort_clus_size_seq = sorted(clustering.clus_size_seq)
    for nclus in num_clusters_range:
        for cluster_list in clustering_ensemble_generator_num(clustering.n_elements, nclus):
            new_clustering= Clustering()
            new_clustering.from_cluster_list(cluster_list)
            if random_model != 'perm' or sorted(new_clustering.clus_size_seq) == sort_clus_size_seq:
                yield new_clustering
    

def clustering_ensemble_generator_num(n_elements, n_clusters):
    """
        A generator for every partition in 'Num', the set of all clusterings with n_elements in n_clusters.

        Based on the solution provided by Adeel Zafar Soomro: `a link`_.

        .. _a link: http://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions

        which was itself based on the algorithm from Knuth:
        (Algorithm U) is described by Knuth in the Art of Computer Programming, Volume 4, Fascicle 3B

        Parameters
        ----------
        n_elements : int
            The number of elements

        n_clusters : int
            The number of clusters

        Returns
        -------
        f : cluster list
            The new clustering as a cluster list.

        >>> import clusim
        >>> for clu in clustering_ensemble_generator_num(n_elements = 5, n_clusters = 3):
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
        for j in xrange(1, n_clusters + 1):
            a[n_elements - n_clusters + j] = j - 1
    return f(n_clusters, n_elements, 0, n_elements, a)