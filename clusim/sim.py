# -*- coding: utf-8 -*-
"""Calculate similarity between clusterings using pair-wise and informaiton theoretic measures.

Example
-------
See the ipython notebook

Notes
-----
    This is just a draft

maybe it works

Attributes
----------
module_level_variable1 : int
    Module level variables may be documented in either the ``Attributes``
    section of the module docstring, or in an inline docstring immediately
    following the variable.

    Either form is acceptable, but the two should not be mixed. Choose
    one convention to document module level variables and be consistent
    with it.


.. _NumPy Documentation HOWTO:
   https://github.com/ajgates42/clusim/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""

from collections import defaultdict, Counter
import numpy as np
import scipy.sparse as spsparse
import mpmath
import itertools

from clusim.clugen import *

available_similarity_measures = ['jaccard_index', 'rand_index', 'fowlkes_mallows_index', 'rogers_tanimoto_index', 'southwood_index', 
'czekanowski_index', 'dice_index', 'sorensen_index', 'pearson_correlation', 'classification_error', 'purity_index',
'fmeasure', 'nmi', 'vi', 'geometric_accuracy', 'overlap_quality', 'onmi', 'omega_index']

available_random_models = ['perm', 'perm1', 'num', 'num1', 'all', 'all1']

def contingency_table(clustering1, clustering2):
    """
        This function creates the contigency table between two clusterings.

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        contigency_table : list of lists
            The clustering1.n_clusters by clustering2.n_clusters contigency table 

        >>> import clusim
        >>> clustering1 = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> cont_table = contingency_table(clustering1, clustering2)
        >>> print cont_table
    """
    assert clustering1.n_elements == clustering2.n_elements

    return [[len(clustering1.clu2elm_dict[clu1] & clustering2.clu2elm_dict[clu2]) for clu2 in clustering2.clusters] for clu1 in clustering1.clusters]

def count_pairwise_cooccurence(clustering1, clustering2):
    """
        This function finds the pairwise cooccurence counts between two clusterings.

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        N11 : int
            The number of element pairs assigned to the same clusters in both clusterings

        N10 : int
            The number of element pairs assigned to the same clusters in clustering1, but 
            different clusters in clustering2

        N01 : int
            The number of element pairs assigned to different clusters in clustering1, but 
            the same clusters in clustering2

        N00 : int
            The number of element pairs assigned to different clusters in both clusterings

        >>> import clusim
        >>> clustering1 = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)
        >>> print_clustering(clustering1)
        >>> print_clustering(clustering2)
        >>> print(N11, "element pairs assigned to the same clusters in both clusterings")
        >>> print(N10, "element pairs assigned to the same clusters in clustering1, but 
            different clusters in clustering2")
        >>> print(N01, "element pairs assigned to different clusters in clustering1, but 
            the same clusters in clustering2")
        >>> print(N00, "element pairs assigned to different clusters in both clusterings")
    """

    cont_tbl = contingency_table(clustering1, clustering2)

    T = np.sum(np.square(cont_tbl))
    R = np.sum(np.square(clustering1.clu_size_seq))
    C = np.sum(np.square(clustering2.clu_size_seq))

    N11 = 0.5 * (T - clustering1.n_elements)
    N10 = 0.5 * (R - T)
    N01 = 0.5 * (C - T)
    N00 = 0.5*clustering1.n_elements *(clustering1.n_elements - 1) - N11 - N10 - N01

    return N11, N10, N01, N00

def entropy(prob_vector, logbase = 2.):
    prob_vector = np.array(prob_vector)
    pos_prob_vector = prob_vector[prob_vector > 0]
    return - np.sum(pos_prob_vector * np.log(pos_prob_vector)/np.log(logbase))

binary_entropy = np.vectorize(lambda p1, p2, logbase = 2: entropy([p1, p2], logbase = logbase))

def hyper(n, a, b, N):
    """ generalized hypergeometric distrubtion """
    return mpmath.binomial(b, n)*mpmath.binomial(N-b,a-n)/mpmath.binomial(N,a)








'''
These are the Pairwise Co-occurence Measures
'''

def jaccard_index(clustering1, clustering2):
    """
        This function calculates the Jaccard index between two clusterings.

        J = N11/(N11+N10+N01)  

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        J : float
            The Jaccard index (between 0.0 and 1.0)

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print(clusim.jaccard_index(clustering1, clustering2) )
    """

    N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)

    denom = N11 + N10 + N01

    # catch the case every element is in its own cluster so denominator is 0
    if denom > 0:
        return N11 / denom
    else:
        return 0.0

def rand_index(clustering1, clustering2):
    """
        This function calculates the Rand index between two clusterings.

        RI = (N11 + N00) / (N11 + N10 + N01 + N00)  

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        RI : float
            The Rand index (between 0.0 and 1.0)

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print(clusim.rand_index(clustering1, clustering2))
    """

    N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)

    return (N11 + N00) / (N11 + N10 + N01 + N00)

def expected_rand_index(n_elements, random_model = 'num', n_clusters1 = 2, n_clusters2 = 2, clu_size_seq1 = None, clu_size_seq2 = None):
    """
        This function calculates the expectation of the Rand index between all pairs of clusterings
        drawn from one of six random models.

        .. note:: Clustering 2 is considered the gold-standard clustering for one-sided expectations

        Parameters
        ----------
        n_elements : int
            The number of elements

        n_clusters1 : int, optional
            The number of clusters in the first clustering

        n_clusters2 : int, optional
            The number of clusters in the second clustering, considered the gold-standard clustering for the one-sided expecations

        clu_size_seq1 : int, optional
            The cluster size seqence of the first clustering

        clu_size_seq2 : int, optional
            The cluster size seqence of the second clustering

        random_model : string
            The random model to use:

            'all' : uniform distrubtion over the set of all clusterings of n_elements

            'all1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements

            'num' : uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'num1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'perm' : the permutation model for a fixed cluster size sequence

            'perm1' : one-sided selction from the permutation model for a fixed cluster size sequence, same as 'perm'

        Returns
        -------
        expected : float
            The expected Rand index (between 0.0 and 1.0)

        >>> import clusim
        >>> print(clusim.expected_rand_index(n_elements = 5, random_model = 'all'))
        >>> print(clusim.expected_rand_index(n_elements = 5, random_model = 'all1', clu_size_seq2 = [1,1,3]))
        >>> print(clusim.expected_rand_index(n_elements = 5, , random_model = 'num', n_clusters1 = 2, n_clusters2 = 3))
        >>> print(clusim.expected_rand_index(n_elements = 5, random_model = 'num1', n_clusters1 = 2, clu_size_seq2 = [1,1,3]))
        >>> print(clusim.expected_rand_index(n_elements = 5, random_model = 'perm', clu_size_seq1 = [2,3], clu_size_seq2 = [1,1,3]))
        >>> print(clusim.expected_rand_index(n_elements = 5, random_model = 'perm1', clu_size_seq1 = [2,3], clu_size_seq2 = [1,1,3]))
    """
    if random_model == 'perm' or random_model == 'perm1':
        npairs = mpmath.binomial(n_elements, 2)
        p = sum([mpmath.binomial(ai, 2) for ai in clu_size_seq1])
        q = sum([mpmath.binomial(bi, 2) for bi in clu_size_seq2])
        expected = 1.0 + 2*p*q / npairs**2 - (p + q) / npairs 

    elif random_model == 'num':
        QA1 = mpmath.stirling2(n_elements-1, n_clusters1) / mpmath.stirling2(n_elements, n_clusters1)
        QB1 = mpmath.stirling2(n_elements-1, n_clusters2) / mpmath.stirling2(n_elements, n_clusters2)
        expected = QA1*QB1 + (1. - QA1)*(1. - QB1)

    elif random_model == 'num1':
        QA1 = mpmath.stirling2(n_elements-1, n_clusters1) / mpmath.stirling2(n_elements, n_clusters1)
        QG1 = sum([mpmath.binomial(gi, 2) for gi in clu_size_seq2])/ mpmath.binomial(n_elements, 2)
        expected = QA1*QG1 + (1. - QA1)*(1. - QG1)

    elif random_model == 'all':
        Q1 = mpmath.bell(n_elements-1)/mpmath.bell(n_elements)
        expected = Q1**2 + (1. - Q1)**2

    elif random_model == 'all1':

        QA1 = mpmath.bell(n_elements-1)/mpmath.bell(n_elements)
        QG1 = sum([mpmath.binomial(gi, 2) for gi in clu_size_seq2])/ mpmath.binomial(n_elements, 2)
        expected = QA1*QG1 + (1. - QA1)*(1. - QG1)

    else:
        ''' TODO: random model not supported'''
        pass

    return np.float64(expected)

def adjrand_index(clustering1, clustering2, random_model = 'perm'):
    """
        This function calculates the adjusted Rand index for one of six random models.

        .. note:: Clustering 2 is considered the gold-standard clustering for one-sided expectations

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering.

        clustering2 : Clustering
            The second clustering.

        random_model : string
            The random model to use:

            'all' : uniform distrubtion over the set of all clusterings of n_elements

            'all1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements

            'num' : uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'num1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'perm' : the permutation model for a fixed cluster size sequence

            'perm1' : one-sided selction from the permutation model for a fixed cluster size sequence, same as 'perm'

        Returns
        -------
        adjusted_rand : float
            The adjusted_rand Rand index

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'all')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'all')
        >>> print(clusim.adjrand_index(clustering1, clustering2, random_model = 'all') )
        >>> print(clusim.adjrand_index(clustering1, clustering2, random_model = 'all1') )
        >>> print(clusim.adjrand_index(clustering1, clustering2, random_model = 'num') )
        >>> print(clusim.adjrand_index(clustering1, clustering2, random_model = 'num1') )
        >>> print(clusim.adjrand_index(clustering1, clustering2, random_model = 'perm') )
        >>> print(clusim.adjrand_index(clustering1, clustering2, random_model = 'perm1') )
    """

    if random_model == 'none':
        exp_rand = 0.0
    else:
        exp_rand = expected_rand_index(n_elements = clustering1.n_elements, 
                                       n_clusters1 = clustering1.n_clusters, 
                                       n_clusters2 = clustering2.n_clusters, 
                                       clu_size_seq1 = clustering1.clu_size_seq, 
                                       clu_size_seq2 = clustering2.clu_size_seq, 
                                       random_model = random_model)

    denom = 1. - exp_rand
    if (denom) > 0:
        return (rand_index(clustering1, clustering2) - exp_rand) / denom
    else:
        return 0.0

def fowlkes_mallows_index(clustering1, clustering2):
    """
        This function calculates the Fowlkes and Mallows index between two clusterings.

        FM = N11 / sqrt( (N11 + N10) * (N11 + N01) )

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        FM : float
            The Fowlkes and Mallows index (between 0.0 and 1.0)

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print(clusim.fowlkes_mallows_index(clustering1, clustering2))
    """
    N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)

    denom = np.sqrt( (N11 + N10) * (N11 + N01) )

    # catch the case every element is in its own cluster so denominator is 0
    if denom > 0:
        return N11 / denom
    else:
        return 0.0

def fmeasure(clustering1, clustering2):
    """
        This function calculates the F-measure between two clusterings.
        
        Also known as: 
        Czekanowski index 
        Dice Symmetric index
        Sorensen index

        F = 2*N11 / (2*N11 + N10 + N01)

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        F : float
            The F-measure (between 0.0 and 1.0)

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print( clusim.fmeasure(clustering1, clustering2))
    """
    N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)

    denom = (2*N11 + N10 + N01)

    # catch the case every element is in its own cluster so denominator is 0
    if denom > 0:
        return 2*N11 / denom
    else:
        return 0.0

def purity_index(clustering1, clustering2):
    """
        This function calculates the Purity index between two clusterings.

        PI = 1/N * 

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        PI : float
            The Purity index (between 0.0 and 1.0)

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print(clusim.purity_index(clustering1, clustering2))
    """

    cont_tbl = contingency_table(clustering1, clustering2)
    return 1.0 / clustering1.n_elements * np.sum(np.max(cont_tbl, axis = np.argmin([clustering2.n_clusters, clustering1.n_clusters])))

def classification_error(clustering1, clustering2):
    """
        This function calculates the Jaccard index between two clusterings.

        CE = 1 - PI

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        CE : float
            The Classification Error (between 0.0 and 1.0)
        
        NOTE - CE is a distance measure, it is 0 for identical clusterings

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print(clusim.classification_error(clustering1, clustering2))
    """
    return 1.0 - purity_index(clustering1, clustering2)

def czekanowski_index(clustering1, clustering2):
    """
        This function calculates the Czekanowski index between two clusterings.

        See Fmeasure
    """
    return fmeasure(clustering1, clustering2)

def dice_index(clustering1, clustering2):
    """
        This function calculates the Dice index between two clusterings.

        See Fmeasure
    """
    return fmeasure(clustering1, clustering2)

def sorensen_index(clustering1, clustering2):
    """
        This function calculates the Sorensen index between two clusterings.

        See Fmeasure
    """
    return fmeasure(clustering1, clustering2)


def rogers_tanimoto_index(clustering1, clustering2):
    """
        This function calculates the Rogers and Tanimoto index between two clusterings.

        RT = (N11 + N00)/(N11 + 2*(N10+N01) + N00) 

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        RT : float
            The Rogers and Tanimoto index (between 0.0 and 1.0)

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print( clusim.rogers_tanimoto_index(clustering1, clustering2) )
    """

    N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)

    return (N11 + N00)/(N11 + 2*(N10+N01) + N00) 

def southwood_index(clustering1, clustering2):
    """ calculate the southwood index 

    N11 / (N10 + N01)

    (cite)

    """

    N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)

    denom = N10 + N01

    # catch the case where the two clusters agree
    if denom > 0:
        return N11 / denom
    else:
        return 1.0


def pearson_correlation(clustering1, clustering2):
    """
        This function calculates the Pearson Correlation between two clusterings.

        PC = (N11 * N00  - N01 * N10) / ( (N11 + N10) * (N11 + N01) * (N00 + N10) * (N00 + N01) ) 

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering

        clustering2 : Clustering
            The second clustering

        Returns
        -------
        PC : float
            The Pearson Correlation (between -1.0 and 1.0)

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'num')
        >>> print(clusim.pearson_correlation(clustering1, clustering2))
    """
    N11, N10, N01, N00 = count_pairwise_cooccurence(clustering1, clustering2)

    numerator = N11 * N00  - N01 * N10
    denom = (N11 + N10) * (N11 + N01) * (N00 + N10) * (N00 + N01) 

    # catch the case every element is in its own cluster so denominator is 0
    if denom > 0:
        return numerator / denom
    else:
        return 0.0

def corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'perm', norm_type = 'sum', nsamples = 100):
    """
        This function calculates the adjusted Similarity for one of six random models.

        .. note:: Clustering 2 is considered the gold-standard clustering for one-sided expectations

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering.

        clustering2 : Clustering
            The second clustering.

        measure : string
            The similarity measure to evalute.  Must be one of the available_similarity_measures.

        random_model : string
            The random model to use:

            'all' : uniform distrubtion over the set of all clusterings of n_elements

            'all1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements

            'num' : uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'num1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'perm' : the permutation model for a fixed cluster size sequence

            'perm1' : one-sided selction from the permutation model for a fixed cluster size sequence, same as 'perm'

        nsamples : int
            The number of random Clusterings sampled to determine the expected similarity.

        Returns
        -------
        adjusted_sim : float
            The adjusted Similarity measure

        >>> import clusim
        >>> clustering1 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'all')
        >>> clustering2 = clusim.make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'all')
        >>> print(clusim.corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'all') )
        >>> print(clusim.corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'all1') )
        >>> print(clusim.corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'num') )
        >>> print(clusim.corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'num1') )
        >>> print(clusim.corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'perm') )
        >>> print(clusim.corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'perm1') )
    """

    # catch the two cases (rand_index or nmi) that can be found 
    if measure == 'rand_index':
        return adjrand_index(clustering1, clustering2, random_model = random_model)

    elif measure == 'nmi':
        return adj_mi(clustering1, clustering2, random_model = random_model, norm_type = norm_type)

    # otherwise, make sure the measure is an available measure
    elif measure in available_similarity_measures:
        if random_model == 'none':
            exp_sim = 0.0
        else:
            exp_sim = sample_expected_sim(clustering1, clustering2, measure = measure, random_model = random_model, nsamples = nsamples)

        denom = 1. - exp_sim
        if (denom) > 0:
            similarity_value = evaluate(str(measure)+'(clustering1, clustering2)')
            return (similarity_value - exp_sim) / denom
        else:
            return 0.0
    else:
        print("Measure %s not supported.  Please choose from one of the available measures:" % measure)
        print(available_similarity_measures)
        return None



def sample_expected_sim(clustering1, clustering2, measure = 'jaccard_index', random_model = 'perm', nsamples = 1, keep_samples = False):
    """
        This function calculates the expected Similarity for all pair-wise comparisons between Clusterings drawn from one of 
        six random models.

        .. note:: Clustering 2 is considered the gold-standard clustering for one-sided expectations

        Parameters
        ----------
        clustering1 : Clustering
            The first clustering.

        clustering2 : Clustering
            The second clustering.

        measure : string
            The similarity measure to evalute.  Must be one of the available_similarity_measures.

        random_model : string
            The random model to use:

            'all' : uniform distrubtion over the set of all clusterings of n_elements

            'all1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements

            'num' : uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'num1' : one-sided selction from the uniform distrubtion over the set of all clusterings of n_elements in n_clusters

            'perm' : the permutation model for a fixed cluster size sequence

            'perm1' : one-sided selction from the permutation model for a fixed cluster size sequence, same as 'perm'

        nsamples : int
            The number of random Clusterings sampled to determine the expected similarity.

        Returns
        -------
        expected_sim : float
            The expected Similarity measure for all pair-wise comparisons under a random model

        >>> import clusim
        >>> clustering1 = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'all')
        >>> clustering2 = make_random_clustering(n_elements = 9, n_clusters = 3, random_model = 'all')
        >>> print(corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'all') )
        >>> print(corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'all1') )
        >>> print(corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'num') )
        >>> print(corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'num1') )
        >>> print(corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'perm') )
        >>> print(corrected_chance(clustering1, clustering2, measure = 'jaccard_index', random_model = 'perm1') )
    """

    # draw nsamples random samples from the random model
    random_clustering1_list = [make_random_clustering(n_elements = clustering1.n_elements, n_clusters = clustering1.n_clusters, 
        clu_size_seq = clustering1.clu_size_seq,
        random_model = random_model, tol = 1.0e-15) for isample in range(nsamples)]

    if '1' in random_model:
        # this is a one-sided model so only compare to the gold-standard clustering
        random_clustering2_list = [clustering2]
    else:
        # this is a two-sided model so draw another nsamples from the random model
        random_clustering2_list = [make_random_clustering(n_elements = clustering2.n_elements, n_clusters = clustering2.n_clusters, 
            clu_size_seq = clustering2.clu_size_seq,
            random_model = random_model, tol = 1.0e-15) for isample in range(nsamples)]

    pairwise_comparisons = [eval(measure + '(c1, c2)') for c1, c2 in itertools.product(random_clustering1_list, random_clustering2_list)]
    
    if keep_samples:
        return pairwise_comparisons
    else:
        return np.mean(pairwise_comparisons)


"""
These are the Information Theoretic Measures
"""
def nmi(clustering1, clustering2, norm_type = 'sum'):
    
    cont_tbl = contingency_table(clustering1, clustering2)

    e1 = entropy(np.array(clustering1.clu_size_seq, dtype = float)/clustering1.n_elements)
    e2 = entropy(np.array(clustering2.clu_size_seq, dtype = float)/clustering2.n_elements)

    e12 = entropy(np.array(cont_tbl, dtype = float)/clustering1.n_elements)

    if norm_type == 'sum':
        normterm = 0.5*(e1 + e2)
    elif norm_type == 'max':
        normterm = max([e1, e2])
    elif norm_type == 'min':
        normterm = min([e1, e2])
    elif norm_type == 'sqrt':
        normterm = np.sqrt(e1*e2)
    elif norm_type == 'none':
        normterm = 1.0

    return (e1 + e2 - e12) / normterm


def vi(clustering1, clustering2, norm_type = 'none'):
    cont_tbl = contingency_table(clustering1, clustering2)

    e1 = entropy(np.array(clustering1.clu_size_seq, dtype = float)/clustering1.n_elements)
    e2 = entropy(np.array(clustering2.clu_size_seq, dtype = float)/clustering2.n_elements)

    e12 = entropy(np.array(cont_tbl, dtype = float)/clustering1.n_elements)
    
    if norm_type == 'entropy':
        return 1.0 - 0.5 * ((e12 - e1)/e2 + (e12 - e2)/e1)
    else:
        return 2 * e12 - e1 - e2

def expected_mi(n_elements, n_clusters1 = 2, n_clusters2 = 2, clu_size_seq1 = None, clu_size_seq2 = None, logbase = 2, random_model = 'num'):

    nf = float(n_elements)

    expected_H1_sum = 0.0
    expected_H2_sum = 0.0
    expected_H12_sum = 0.0

    symmetric_sum = True

    # find the counts for clustering 1
    if 'perm' in random_model:
        counter1 = Counter(clu_size_seq1)
        cluster_size_range1 = sorted(set(clu_size_seq1))
        cluster_counts1 = [counter1[clu] for clu in cluster_size_range1]

    elif 'num' in random_model:
        sn1 = mpmath.stirling2(n_elements, n_clusters1)
        cluster_size_range1 = range(1, n_elements + 1 - (n_clusters1 - 1))
        cluster_counts1 = [mpmath.binomial(n_elements, ai)*mpmath.stirling2(n_elements - ai, n_clusters1 - 1)/sn1 for ai in cluster_size_range1]

    elif 'all' in random_model:
        bn = mpmath.bell(n_elements)
        cluster_size_range1 = range(1, n_elements + 1)
        cluster_counts1 = [mpmath.binomial(n_elements, ai)*mpmath.bell(n_elements - ai)/bn for ai in range(1, n_elements + 1)]

    # find the counts for clustering 2
    if random_model in ['perm', 'perm1', 'num1', 'all1']:
        counter2 = Counter(clu_size_seq2)
        cluster_size_range2 = sorted(set(clu_size_seq2))
        cluster_counts2 = [counter2[clu] for clu in cluster_size_range2]
        symmetric_sum = False
        for jclus, clu_size2 in enumerate(cluster_size_range2):
            expected_H2_sum += clu_size2/nf*mpmath.log(clu_size2/nf)*cluster_counts2[jclus]

    elif random_model == 'num' and n_clusters1 != n_clusters2:
        sn2 = mpmath.stirling2(n_elements, n_clusters2)
        cluster_size_range2 = range(1, n_elements + 1 - (n_clusters2 - 1))
        cluster_counts2 = [mpmath.binomial(n_elements, bj)*mpmath.stirling2(n_elements - bj, n_clusters2 - 1)/sn2 for bj in cluster_size_range2]
        symmetric_sum = False
        for jclus, clu_size2 in enumerate(cluster_size_range2):
            expected_H2_sum += clu_size2/nf*mpmath.log(clu_size2/nf)*cluster_counts2[jclus]

    else:
        cluster_size_range2 = cluster_size_range1
        cluster_counts2 = cluster_counts1

        
    if symmetric_sum:
        for iclus, clu_size1 in enumerate(cluster_size_range1):

            expected_H1_sum += clu_size1/nf*mpmath.log(clu_size1/nf)*cluster_counts1[iclus]
            
            for jclus in range(iclus):
                clu_size2 = cluster_size_range1[jclus]
                
                for nij in range(max(clu_size1 + clu_size2 - n_elements, 1), clu_size2 + 1):
                    expected_H12_sum += 2*cluster_counts1[iclus]*cluster_counts2[jclus]*hyper(nij, clu_size1, clu_size2, n_elements)* nij/nf*mpmath.log(nij/nf)

            for nij in range(max(2*clu_size1 - n_elements, 1), clu_size1 + 1):
                expected_H12_sum += cluster_counts1[iclus]**2 *hyper(nij, clu_size1, clu_size1, n_elements)* nij/nf*mpmath.log(nij/nf)
        expected_H2_sum = expected_H1_sum

    else:
        for iclus, clu_size1 in enumerate(cluster_size_range1):

            expected_H1_sum += clu_size1/nf*mpmath.log(clu_size1/nf)*cluster_counts1[iclus]
            
            for jclus, clu_size2 in enumerate(cluster_size_range2):
                
                for nij in range(max(clu_size1 + clu_size2 - n_elements, 1), min(clu_size1, clu_size2) + 1):
                    expected_H12_sum += cluster_counts1[iclus]*cluster_counts2[jclus]*hyper(nij, clu_size1, clu_size2, n_elements)* nij/nf*mpmath.log(nij/nf)

            

    expected_H1_sum /= -mpmath.log(logbase)
    expected_H2_sum /= -mpmath.log(logbase)
    expected_H12_sum /= -mpmath.log(logbase)
    return expected_H1_sum + expected_H2_sum - expected_H12_sum

def adj_mi(clustering1, clustering2, random_model = 'perm', norm_type = 'none', logbase = 2):
    """ calculate the adjusted Mutual Information for all random models

    (cite)

    """

    if random_model == 'none':
        exp_mi = 0.0
    else:
        exp_mi = expected_mi(n_elements = clustering1.n_elements, 
                                       n_clusters1 = clustering1.n_clusters, 
                                       n_clusters2 = clustering2.n_clusters, 
                                       clu_size_seq1 = clustering1.clu_size_seq, 
                                       clu_size_seq2 = clustering2.clu_size_seq, 
                                       random_model = random_model,
                                       logbase = logbase)

    if random_model == 'perm':
        e1 = entropy(np.array(clustering1.clu_size_seq, dtype = float)/clustering1.n_elements, logbase = logbase)
        e2 = entropy(np.array(clustering2.clu_size_seq, dtype = float)/clustering2.n_elements, logbase = logbase)

    elif random_model == 'num':
        e1 = np.log(clustering1.n_clusters) / np.log(logbase)
        e2 = np.log(clustering2.n_clusters) / np.log(logbase)

    elif random_model == 'all':
        e1 = np.log(clustering1.n_elements) / np.log(logbase)
        e2 = np.log(clustering2.n_elements) / np.log(logbase)


    if norm_type == 'sum':
        normterm = 0.5*(e1 + e2) - exp_mi
    elif norm_type == 'max':
        normterm = max([e1, e2]) - exp_mi
    elif norm_type == 'min':
        normterm = min([e1, e2]) - exp_mi
    elif norm_type == 'sqrt':
        normterm = np.sqrt(e1*e2) - exp_mi
    elif norm_type == 'none':
        normterm = 1.0

    return (nmi(clustering1, clustering2, norm_type = 'none') - exp_mi) / normterm




"""
These are for overlapping clusterings
"""

def geometric_accuracy(clustering1, clustering2):
    ''' Nepusz et al. (2012) Nature Methods 9, 471 - 472'''

    cont_tbl = contingency_table(clustering1, clustering2)

    Nclusters = np.sum(clustering1.clu_size_seq)
    Mclusters = np.sum(cont_tbl)

    Sn = np.sum(np.max(cont_tbl, axis = 1))/float(Nclusters)

    PPV = np.sum(np.max(cont_tbl, axis = 0))/float(Mclusters)

    return np.sqrt(Sn * PPV)

def overlap_quality(clustering1, clustering2):
    ''' Ahn et al. (2010) Nature'''
    num_memberships1 = [len(clustering1.elm2clu_dict[el]) for el in clustering1.elements]
    num_memberships2 = [len(clustering2.elm2clu_dict[el]) for el in clustering2.elements]

    overlap_dist = np.zeros((max(num_memberships1) + 1, max(num_memberships2) + 1), dtype = float)
    for i_el in range(clustering1.n_elements):
        overlap_dist[num_memberships1[i_el], num_memberships2[i_el]] += 1
    
    overlap_dist = overlap_dist/np.sum(overlap_dist)
    
    return entropy(np.sum(overlap_dist, axis = 0)) + entropy(np.sum(overlap_dist, axis = 1)) + entropy(overlap_dist)

def onmi(clustering1, clustering2):
    '''
    Overlapping Normalized Mutual Information

    (cite)
    '''

    cont_tbl = contingency_table(clustering1, clustering2)

    e12 = entropy(np.array(cont_tbl, dtype = float)/clustering1.n_elements)

    prob_clu1 = np.array(clustering1.clu_size_seq, dtype = float)/clustering1.n_elements
    prob_clu2 = np.array(clustering2.clu_size_seq, dtype = float)/clustering2.n_elements
    joint_prob = np.array(cont_tbl, dtype = float)/clustering1.n_elements

    entropy_rv1 = binary_entropy(prob_clu1, 1.0-prob_clu1)
    entropy_rv2 = binary_entropy(prob_clu2, 1.0-prob_clu2)

    entropy_rv12_pure = binary_entropy(joint_prob, 1.0-joint_prob)

    entropy_rv12_mixing = binary_entropy(prob_clu1.reshape(clustering1.n_clusters, 1) - joint_prob, 
        prob_clu2.reshape(1, clustering2.n_clusters)-joint_prob)
    
    minimize_conditions_rv12 = entropy_rv12_pure > entropy_rv12_mixing

    e_r2_cond_r1 = (entropy_rv12_pure + entropy_rv12_mixing).T - entropy_rv1

    e_r2_cond_Boldr1 = np.zeros(clustering2.n_clusters)
    for k2 in range(clustering2.n_clusters):
        if np.any(minimize_conditions_rv12.T[k2]):
            e_r2_cond_Boldr1[k2] = e_r2_cond_r1[k2,minimize_conditions_rv12.T[k2]].min() / entropy_rv2[k2]
        else:
            e_r2_cond_Boldr1[k2] = 1.0

    

    e_r1_cond_r2 = (entropy_rv12_pure + entropy_rv12_mixing) - entropy_rv2

    e_r1_cond_Boldr2 = np.zeros(clustering1.n_clusters)
    for k1 in range(clustering1.n_clusters):
        if np.any(minimize_conditions_rv12[k1]):
            e_r1_cond_Boldr2[k1] = e_r1_cond_r2[k1,minimize_conditions_rv12[k1]].min() / entropy_rv1[k1]
        else:
            e_r1_cond_Boldr2[k1] = 1.0

    
    return 1.0 - 0.5*(np.mean(e_r1_cond_Boldr2) + np.mean(e_r2_cond_Boldr1))

def make_overlapping_membership_matrix(clustering):
    '''
    the overlapping membership matrix needed for the Omega index
    '''

    A = spsparse.csr_matrix((clustering.n_elements, clustering.n_elements), dtype='int')
    for clu in clustering.clusters:
        v = np.zeros(clustering.n_elements)
        v[list(clustering.clu2elm_dict[clu])] = 1
        v = spsparse.csr_matrix(v, dtype ='int')
        A += v.T*v
    return A

def omega_index(clustering1, clustering2):
    '''
    Omega index

    (cite)
    '''

    A1 = make_overlapping_membership_matrix(clustering1)
    A2 = make_overlapping_membership_matrix(clustering2)

    M = clustering1.n_elements * (clustering1.n_elements - 1) / 2.0

    maxNover = max(max(A1.diagonal()), max(A2.diagonal())) + 1

    Anot = spsparse.triu((A1 != A2), k = 1).sum()

    omega_u = 1.0 - Anot.sum() / M

    t_0_1 = M - spsparse.triu((A1 != 0), k = 1).sum()
    t_0_2 = M - spsparse.triu((A2 != 0), k = 1).sum()

    t_k_1 = [spsparse.triu((A1 == i), k = 1).sum() for i in range(1, maxNover)]
    t_k_2 = [spsparse.triu((A2 == i), k = 1).sum() for i in range(1, maxNover)]

    omega_e = (t_0_1*t_0_2 + np.dot(t_k_1, t_k_2) ) / M**2

    return (omega_u - omega_e) / (1.0 - omega_e)
