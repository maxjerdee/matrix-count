# Examples of typical usage of the matrix_count module

import matrix_count as mc

# Margin of a 8x8 symmetric non-negative integer matrix with even diagonal entries
margin = [1, 1, 1, 1, 3, 3]

# Estimate the logarithm of the number of symmetric matrices with given margin sum
# (number of multigraphs with given degree sequence)
estimate = mc.estimate_log_symmetric_matrices(margin, alpha=1, binary_matrix=True)
print("Estimated log count of symmetric matrices:", estimate)

# Count the number of such matrices
count, count_err = mc.count_log_symmetric_matrices(
    margin, alpha=1 , binary_matrix=True
)
print("Log count of symmetric matrices:", count, "+/-", count_err)

# Sample from the space of such matrices
num_samples = 3
for _t in range(num_samples):
    sample, entropy = mc.sample_symmetric_matrix(margin, binary_matrix=True)
    print("Sampled matrix:")
    print(sample)
    print("Minus log probability of sampled matrix:", entropy)


from matrix_count import _util

eps = 0.00001

print(_util._log_binom_c(-8,-14))
print(_util._lgamma_c(-8 + 1 + eps))
print(_util._lgamma_c(-14 + 1 + eps))
print(_util._lgamma_c(-8 - (-14) + 1 + eps))
print(_util._lgamma_c(-8 + 1 + eps) - _util._lgamma_c(-14 + 1 + eps) - _util._lgamma_c(-8 - (-14) + 1 + eps))


