# Example of using matrix-count to analytically estimate the number of matrices with given margins (and further conditions)

# Arbitrary matrices
row_margin = [220, 215, 93, 64]
column_margin = [108, 286, 71, 127]


# from matrix_count.estimate import *
import matrix_count
import numpy as np

import matplotlib.pyplot as plt

# test_margin = [20,11,3,4]
# test_margin = [20,11,3,4,3,3,3,3]
test_margin = [20,11,3]
num_samples = 10000

true_count = 34
# true_count = None

# Estimate the number of matrices with given margins
estimate = matrix_count.estimate_log_symmetric_matrices(test_margin, estimate_order=2)

estimate_3 = matrix_count.estimate_log_symmetric_matrices(test_margin, estimate_order=3)

logEs = []
log_count_err_ests = []

entropies = []
for t in range(num_samples):
    sample, entropy = matrix_count.sample_symmetric_matrix(test_margin)
    entropies.append(entropy)
    # log(Delta log E) = log(Delta E/E) = 1/2log(E2 - E^2) - 1/2 log(n) - log(E)
    logE2 = matrix_count._log_sum_exp(2*np.array(entropies)) - np.log(len(entropies))
    logE = matrix_count._log_sum_exp(entropies) - np.log(len(entropies))
    log_std = 0.5 * (np.log(np.exp(0) - np.exp(2*logE - logE2)) + logE2)
    log_count_err_est = np.exp(log_std - 0.5 * np.log(len(entropies)) - logE)
    logEs.append(logE)
    log_count_err_ests.append(log_count_err_est)

entropies = np.array(entropies)

plot_frequency = 100 # Number of samples between which to plot the result

logEs = np.array(logEs)
log_count_err_ests = np.array(log_count_err_ests)
inds = np.arange(len(logEs), step=plot_frequency)
plt.errorbar(inds,logEs[inds], yerr=log_count_err_ests[inds], label="SIS estimate")
plt.plot(inds,estimate*np.ones(len(inds)), label="Analytical estimate")
plt.plot(inds,estimate_3*np.ones(len(inds)), label="Analytical estimate (3rd)")
if true_count is not None:
    plt.plot(inds,np.log(true_count)*np.ones(len(inds)), label="True count")
plt.xlabel("Number of samples")
plt.ylabel("Log count")
plt.legend()
plt.savefig("test_error.png")