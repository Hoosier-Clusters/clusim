.. CluSim documentation master file, created by
   sphinx-quickstart on Thu Aug 23 00:00:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CluSim's documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:


Examples and Usage
===================


The “Clustering”
=================
.. autoclass:: clusim.clustering.Clustering
    :members:

.. autoclass:: clusim.clustering.ClusterError

.. automodule:: clusim.clustering
    :members:


Clustering Generation
======================
.. automodule:: clusim.clugen
   :members:


Clustering Similarity
======================
The different clustering similarity measures available.


Pairwise Counting Measures
--------------------------
.. automodule:: clusim.sim
   :members: contingency_table, count_pairwise_cooccurence, jaccard_index, rand_index, fowlkes_mallows_index, fmeasure, purity_index, classification_error, czekanowski_index, dice_index, sorensen_index, rogers_tanimoto_index, southwood_index, pearson_correlation


Information Theoretic Measures
------------------------------
..autofunction:: nmi, vi
   :members: nmi, vi


Correction for Chance
------------------------------
..automodule:: clusim.sim
   :members: corrected_chance, sample_expected_sim, expected_rand_index, adjrand_index, adj_mi, expected_mi


Overlapping Clustering Similarity
---------------------------------
clussim.sim :members: geometric_accuracy, overlap_quality, onmi, omega_index






Element-centric Clustering Similarity
======================================
.. automodule:: clusim.clusimelement
   :members:




DAG and Dendrogram
====================
.. automodule:: clusim.dag
   :members:



References
===========
.. bibliography:: clusimref.bib
