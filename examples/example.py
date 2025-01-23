from __future__ import annotations

import matrix_count

margin = [20, 11, 3]
estimate = matrix_count.estimate_log_symmetric_matrices(margin, verbose=True, alpha=1)
print("Estimated log count of symmetric matrices:", estimate)

num_samples = 1000
samples = []
entropies = []

for _t in range(num_samples):
    sample, entropy = matrix_count.sample_symmetric_matrix(margin)
    samples.append(sample)
    entropies.append(entropy)

count, count_err = matrix_count.count_log_symmetric_matrices(
    margin, verbose=True, alpha=1
)
print("Log count of symmetric matrices:", count, "+/-", count_err)
