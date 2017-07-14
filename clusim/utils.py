""" useful functions for clustering similarity """
import collections
from six import iteritems

def dict_intersection(dict_list):
    """ given a list of dictionaries, return a list of dictionaries with only the
    common keys. """

    common_keys = set.intersection(*[set(d) for d in dict_list])
    return [{k:v for k, v in iteritems(d) if k in common_keys} for d in dict_list]

def mem_dict_union(mem_dict_list, label_modifiers=None):
    """ TODO: unclear what this function is doing. """

    if not label_modifiers:
        label_modifiers = [chr(ord('a') + dict_idx)
                           for dict_idx, d in enumerate(mem_dict_list)]

    new_mem_dict = collections.defaultdict(list)
    for dict_idx, mem_dict in enumerate(mem_dict_list):
        for node, clusters in iteritems(mem_dict):
            for cluster in clusters:
                # TODO: what does this filter do? does it necessary?
                base_comm_names = [''] + filter(None, cluster.split('.'))
                modifier = '.{}'.format(label_modifiers[dict_idx])
                new_comm_name = modifier.join(base_comm_names)
                new_mem_dict[node].append(new_comm_name)