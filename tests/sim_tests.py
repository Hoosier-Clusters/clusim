# -*- coding: utf-8 -*-
#
# Tests for ``sim.py``
# These tests were hand calculated by Alexander J. Gates: ajgates42@gmail.com
#

import clusim
from numpy.testing import assert_approx_equal
from numpy import mean

def test_comparison_example():
	c1_elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[0], 4:[2], 5:[1]}
	c2_elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[0], 4:[2], 5:[2]}

	c1 = clusim.Clustering(elm2clu_dict = c1_elm2clu_dict)
	c2 = clusim.Clustering(elm2clu_dict = c2_elm2clu_dict)

	N11, N10, N01, N00 = clusim.count_pairwise_cooccurence(c1, c2)

	assert N11 == 2, "Element Co-occurance counts for N11 does not match. %s != %s" % (N11, 2)
	assert N10 == 2, "Element Co-occurance counts for N10 does not match. %s != %s" % (N10, 2)
	assert N01 == 1, "Element Co-occurance counts for N01 does not match. %s != %s" % (N01, 1)
	assert N00 == 10, "Element Co-occurance counts for N00 does not match. %s != %s" % (N00, 10)

	known_sim_values = {'jaccard_index':0.4, 'rand_index':0.8 , 'fowlkes_mallows_index':0.5773502691896258, 
	'rogers_tanimoto_index':2./3., 'southwood_index':2./3., 'czekanowski_index':0.5714285714285714,
	'dice_index':0.5714285714285714, 'sorensen_index':0.5714285714285714, 'pearson_correlation':0.011363636363636364,
	'classification_error':0.16666666666666674, 'purity_index':0.8333333333333333, 'fmeasure':0.5714285714285714,
	'nmi':0.7396673768007593, 'vi':0.792481250360578, 'geometric_accuracy':0.8333333333333334, 'overlap_quality':-0.0,
	'onmi':0.7449589906475155, 'omega_index':0.44444444444444453}
	
	for simfunc in clusim.available_similarity_measures:
		simvalue = eval('clusim.' + simfunc+'(c1, c2)')
		assert  simvalue == known_sim_values[simfunc], "Similarity Measure %s does not match. %s != %s" % (simfunc, simvalue, known_sim_values[simfunc])


def test_model_example():
	c1_elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[0]}
	c2_elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[1]}

	c1 = clusim.Clustering(elm2clu_dict = c1_elm2clu_dict)
	c2 = clusim.Clustering(elm2clu_dict = c2_elm2clu_dict)

	known_rand_values = {'perm': 0.5, 'perm1':0.5, 'num':0.510204081632653, 'num1':0.5 , 'all':0.555555555555556, 'all1':0.5}

	known_mi_values = {'perm':0.311278124459133, 'perm1':0.311278124459133, 'num':0.309927805548467, 'num1':0.301825892084476,
				'all':0.611635721962606, 'all1':0.419448541053684}

	for rdm in clusim.available_random_models:
		exp_rand_value = clusim.expected_rand_index(n_elements = c1.n_elements, 
                                       n_clusters1 = c1.n_clusters, 
                                       n_clusters2 = c2.n_clusters, 
                                       clu_size_seq1 = c1.clu_size_seq, 
                                       clu_size_seq2 = c2.clu_size_seq, 
                                       random_model = rdm)
		assert_approx_equal(exp_rand_value, known_rand_values[rdm], 10**(-10), "Expected Rand Index with %s Random Model does not match. %s != %s" % (rdm, exp_rand_value, known_rand_values[rdm]))
		#print(exp_rand_value, type(exp_rand_value), known_rand_values[rdm], type(known_rand_values[rdm]))
		#assert   == known_rand_values[rdm], "Expected Rand Index with %s Random Model does not match. %s != %s" % (rdm, exp_rand_value, known_rand_values[rdm])

		exp_mi_value = float(clusim.expected_mi(n_elements = c1.n_elements, 
                                       n_clusters1 = c1.n_clusters, 
                                       n_clusters2 = c2.n_clusters, 
                                       clu_size_seq1 = c1.clu_size_seq, 
                                       clu_size_seq2 = c2.clu_size_seq, 
                                       random_model = rdm,
                                       logbase = 2.))
		assert_approx_equal(exp_mi_value, known_mi_values[rdm], 10**(-10), "Expected MI with %s Random Model does not match. %s != %s" % (rdm, exp_mi_value, known_mi_values[rdm]) )

def test_elementsim_example():

	# taken from Fig 3 of Gates et al (2018) Scientific Reports
	
	# overlapping clustering
	c1_elm2clu_dict = {0:[0], 1:[0], 2:[0], 3:[3], 4:['.3'], 5:['.3', '.9'], 6:['.9']}

	# hierarchical clustering
	c2_elm2clu_dict = {0:[1], 1:[1], 2:[2], 3:[5], 4:[5], 5:[6, 8], 6:[9]}
	c2_dag = clusim.DAG()
	c2_dag.add_edges_from([(0,1), (0,2), (3,4), (4,5), (4,6), (3,7), (7,8), (7,9)])

	c1 = clusim.Clustering(elm2clu_dict = c1_elm2clu_dict)
	c2 = clusim.Clustering(elm2clu_dict = c2_elm2clu_dict, hier_graph = c2_dag)

	known_elsim = [0.92875658, 0.92875658, 0.85751315, 0.25717544, 0.74282456, 0.82083876, 0.80767074]

	elsim, ellabels = clusim.element_sim_elscore(c1, c2, alpha = 0.9, r = 1., r2 = None, rescale_path_type = 'max')

	for i in range(7):
		assert_approx_equal(elsim[i], known_elsim[i], 10**(-10), "Element-centric similarity for element %s does not match. %s != %s" % (i, elsim[i], known_elsim[i]) )


if __name__ == "__main__":

    test_comparison_example()

    test_model_example()

    test_elementsim_example()

