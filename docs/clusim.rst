.. CluSim documentation master file, created by
   sphinx-quickstart on Thu Aug 23 00:00:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CluSim's documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:


Installation
===================
This package (will be) available in PyPI. Just run the following command on terminal to install.

>>> pip install clusim

You can also source the code directly from the github [project page](https://github.com/Hoosier-Clusters/clusim).


Examples and Usage
===================

A first comparison
--------------------------
We start by importing the required modules

>>> from clusim.clustering import Clustering, print_clustering
>>> import clusim.sim as sim

The simplest way to make a Clustering is to use an elm2clu_dict which maps each element.

>>> c1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
>>> c2 = Clustering(elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[1], 4:[1], 5:[2]})

>>> print_clustering(c1)
>>> print_clustering(c2)

Finally, the similarity of the two Clusterings can be found using the Jaccard Index.

>>> sim.jaccard_index(c1, c2)


Basics of element-centric similarity
--------------------------------------
>>> from clusim.clustering import Clustering, print_clustering
>>> import clusim.sim as sim

>>> c1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
>>> c2 = Clustering(elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[1], 4:[1], 5:[2]})

The basic element-centric similarity score with a fixed alpha:

>>> sim.element_sim(c1, c2, alpha = 0.9)

We can also get the element scores.  Note that since non-numberic elements are allowed, the element scores returns a dict which maps the elements to the index in the elementScore array.

>>> elementScores, relabeled_elements = sim.element_sim_elscore(c1, c2, alpha = 0.9)


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
.. autofunction:: clusim.sim.contingency_table

.. autofunction:: clusim.sim.count_pairwise_cooccurence
.. autofunction:: clusim.sim.jaccard_index
.. autofunction:: clusim.sim.rand_index
.. autofunction:: clusim.sim.fowlkes_mallows_index
.. autofunction:: clusim.sim.fmeasure
.. autofunction:: clusim.sim.purity_index
.. autofunction:: clusim.sim.classification_error
.. autofunction:: clusim.sim.czekanowski_index
.. autofunction:: clusim.sim.dice_index
.. autofunction:: clusim.sim.sorensen_index
.. autofunction:: clusim.sim.rogers_tanimoto_index
.. autofunction:: clusim.sim.southwood_index
.. autofunction:: clusim.sim.pearson_correlation


Information Theoretic Measures
------------------------------
.. autofunction:: clusim.sim.nmi
.. autofunction:: clusim.sim.vi


Correction for Chance
------------------------------
.. autofunction:: clusim.sim.corrected_chance
.. autofunction:: clusim.sim.sample_expected_sim
.. autofunction:: clusim.sim.expected_rand_index
.. autofunction:: clusim.sim.adjrand_index
.. autofunction:: clusim.sim.adj_mi
.. autofunction:: clusim.sim.expected_mi


Overlapping Clustering Similarity
----------------------------------
.. autofunction:: clusim.sim.onmi
.. autofunction:: clusim.sim.omega_index
.. autofunction:: clusim.sim.geometric_accuracy
.. autofunction:: clusim.sim.overlap_quality



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
