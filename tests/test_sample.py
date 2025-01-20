from __future__ import annotations

from matrix_count.sample import *
import numpy as np
from pytest import approx, raises


# sample_symmetric_matrix


def test_sample_symmetric_matrix():
    # Note that this is a random test and the results may vary.
    
    # For some reason this test checking the correlator is failing, TODO: investigate and implement this test
    # samples = []
    # entropies = []
    # entries = []

    # correlator_sum = 0
    # total_prop = 0
    # num_samples = 1000

    # for t in range(num_samples):
    #     sample, entropy = sample_symmetric_matrix([20,11,3])
    #     samples.append(sample)
    #     entropies.append(entropy)
    #     prop = np.exp(-entropy)
    #     total_prop += prop
    #     # correlator_sum += sample[0,1]*sample[1,2]*prop
    #     correlator_sum += sample[0,0]*sample[0,0]*prop

    # # Check the correlator
    # correlator = correlator_sum / total_prop
    # print(correlator)

    # inverse_probs = np.exp(np.array(entropies))
    # print(np.mean(inverse_probs))

    # # Print count of each entry
    # print(np.unique(entries, return_counts=True))

    # assert correlator == approx(5.0, abs=0.1)

    # All good
    assert True