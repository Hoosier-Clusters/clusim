# -*- coding: utf-8 -*-
#
# These tests were designed by Jisung Yoon - jisung.yoon@postech.ac.kr
# Note that the implementation code use threshold as 0.0001, so we use a significant level as 3.
# If reduce the threshold, then results will be more precise, but takes some time.
# Also, prpack is kind of approximated method.

import json
import time

from numpy.testing import assert_approx_equal

import clusim.sim as sim
from clusim.clustering import Clustering

def test_simple_example():
    c1_elm2clu_dict = {0: [0, 1], 1: [1, 2], 2: [1, 3], 3: [0], 4: [2], 5: [1]}
    c2_elm2clu_dict = {0: [0], 1: [1], 2: [1], 3: [0, 3], 4: [2, 4], 5: [2]}

    c1 = Clustering(elm2clu_dict=c1_elm2clu_dict)
    c2 = Clustering(elm2clu_dict=c2_elm2clu_dict)

    sim_ppr_pack = sim.element_sim(
        c1,
        c2,
        alpha=0.9,
        r=1.0,
        r2=None,
        rescale_path_type="max",
        ppr_implementation="prpack",
    )
    sim_ppr_power_iteration = sim.element_sim(
        c1,
        c2,
        alpha=0.9,
        r=1.0,
        r2=None,
        rescale_path_type="max",
        ppr_implementation="power_iteration",
    )

    assert_approx_equal(sim_ppr_pack, sim_ppr_power_iteration, significant=3)


def test_real_example_on_overlapping_community():
    ground_truth_community = json.load(
        open("ground_truth_community_Philosophy.json", "r")
    )
    detected_community = json.load(open("detected_community_Philosophy.json", "r"))

    c1 = Clustering(elm2clu_dict=ground_truth_community)
    c2 = Clustering(elm2clu_dict=detected_community)

    start = time.time()
    sim_ppr_pack = sim.element_sim(
        c1,
        c2,
        alpha=0.9,
        r=1.0,
        r2=None,
        rescale_path_type="max",
        ppr_implementation="prpack",
    )
    end = time.time()
    print("prpack elapsed time: {}s".format(end - start))
    start = time.time()
    sim_ppr_power_iteration = sim.element_sim(
        c1,
        c2,
        alpha=0.9,
        r=1.0,
        r2=None,
        rescale_path_type="max",
        ppr_implementation="power_iteration",
    )
    end = time.time()
    print("power iteration elapsed time: {}s".format(end - start))

    assert_approx_equal(sim_ppr_pack, sim_ppr_power_iteration, significant=3)


if __name__ == "__main__":

    test_simple_example()
    test_real_example_on_overlapping_community()
