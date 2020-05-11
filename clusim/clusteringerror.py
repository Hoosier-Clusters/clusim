# -*- coding: utf-8 -*-
"""
.. module:: Clustering Errors
    :synopsis: Common erros and abuse cases.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

class ClusteringError(Exception):
    """
    Base Class for clustering errors.
    """

class EmptyClusteringError(ClusteringError):
    """
    Raised when the clustering does not contain elements.
    """
    def __str__(self):
        return 'Clustering must have one or more elements in one or more clusters.'


class InvalidElementError(ClusteringError):
    """
    Raised when an element is None or np.nan.
    """
    def __str__(self):
        return 'Elements cannot be None or NaN.'

class InvalidClusterError(ClusteringError):
    """
    Raised when an element is None or np.nan.
    """
    def __str__(self):
        return 'Clusters cannot be None or NaN.'

class EmptyClusterError(ClusteringError):
    """
    Raised when a clustering has empty clusters.
    """

    def __init__(self, n_emptys = None, empty_list = None):
        self.n_emptys = n_emptys
        self.empty_list = empty_list

    def __str__(self):
        if self.empty_list:
            return 'The following clusters contain 0 elements:\n{}'.format(self.empty_list)
        elif self.n_emptys:
            return 'There are {} clusters with 0 elements.'.format(self.n_emptys)
        else:
            return 'EmptyClusterError'

class UnassignedElementError(ClusteringError):
    """
    Raised when elements are not assigned to clusters.
    """

    def __init__(self, n_unassigned = None, unassigned_list = None):
        self.n_unassigned = n_unassigned
        self.unassigned_list = unassigned_list

    def __str__(self):
        if self.unassigned_list:
            return 'The following elements were not assigned a cluster:\n{}'.format(self.unassigned_list)
        elif self.n_unassigned:
            return 'There are {} elements unassigned to a cluster.'.format(self.n_unassigned)
        else:
            return 'UnassignedElementError'



class ClusteringSimilarityError(Exception):
    """
    Base Class for clustering similarity errors.
    """
    def __str__(self):
        return 'The element sets must be the same for both clusterings.'


