# Doing some tests

import matrix_count

test_margin = [2,2,2,2,2,2,2,2]

print(matrix_count.estimate_log_symmetric_matrices(test_margin, binary_matrix=True, estimate_order=2))
print(matrix_count.count_log_symmetric_matrices(test_margin, binary_matrix=True))

print(matrix_count.estimate_log_symmetric_matrices(test_margin, binary_matrix=False, estimate_order=2))
print(matrix_count.count_log_symmetric_matrices(test_margin, binary_matrix=False))

large_margin = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
print(matrix_count.estimate_log_symmetric_matrices(large_margin, binary_matrix=True, estimate_order=2))
print(matrix_count.count_log_symmetric_matrices(large_margin, binary_matrix=True))
print(matrix_count.estimate_log_symmetric_matrices(large_margin, binary_matrix=False, estimate_order=3))
print(matrix_count.count_log_symmetric_matrices(large_margin, binary_matrix=False))