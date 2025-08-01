# Checking that the reported errors are reasonable
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import matrix_count as mc

test_margin = [3, 3, 3, 3, 2, 2]
true_count = np.exp(7.51098)

est_counts = []
est_errs = []
z_scores = []

num_trials = 250

for _trial_num in range(num_trials):
    (log_count_est, log_count_est_err) = mc.count_log_symmetric_matrices(
        test_margin
    )
    est_counts.append(log_count_est)
    est_errs.append(log_count_est_err)
    z_scores.append((log_count_est - np.log(true_count)) / log_count_est_err)

# Plot a histogram of the z-scores
plt.clf()
plt.hist(z_scores, bins=20)
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.savefig("errors.png")

# Can observe that the z scores are roughly distributed as a unit Gaussian with mean 0 and standard deviation 1 as we would expect
print("Mean of the z-scores:", np.mean(z_scores))
print("Standard deviation of the z-scores:", np.std(z_scores))

# Mean of the z-scores: 0.004734026408860607
# Standard deviation of the z-scores: 0.9801903451557386

# TODO: include a case where the reported error is too small, as can occur if the errors are highly non-Gaussian
