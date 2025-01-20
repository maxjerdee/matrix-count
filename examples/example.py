
import matrix_count
import numpy as np

# Margin of a 8x8 symmetric non-negative integer matrix with even diagonal entries
margin = [10,9,8,7,6,5,4,3]
# margin = [20,11,3]

# Estimate the logarithm of the number of symmetric matrices with given margin sum
estimate = matrix_count.estimate_log_symmetric_matrices(margin,verbose=True,alpha=1)
print("Estimated log count of symmetric matrices:", estimate)

for t in range(1000):
    sample, entropy = matrix_count.sample_symmetric_matrix(margin)
    print(np.sum(sample,axis=0))
    print(entropy)

# Count the number of such matrices
count, count_err = matrix_count.count_log_symmetric_matrices(margin,verbose=True,alpha=1)
print("Log count of symmetric matrices:", count, "+/-", count_err)