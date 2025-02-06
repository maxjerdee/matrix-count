# Doing some tests

import matrix_count
import numpy as np

def erdos_gallai_check_parts(ks) -> bool:
    """Check if a sequence is graphical using the Erdos-Gallai theorem

    :param ks: Sequence of non-negative integers
    :type ks: ArrayLike
    :return: Whether the sequence is graphical
    :rtype: bool
    """
    ks = np.array(ks)
    if np.sum(ks) % 2 == 1 or np.any(ks < 0) or np.any(ks > len(ks) - 1):
        return False
    ks = -np.sort(-ks)
    print(ks)
    for l_val in range(1, len(ks) + 1):
        left: int = np.sum(ks[:l_val])
        right: int = l_val * (l_val - 1)
        for _ in range(l_val, len(ks)):
            right += min(l_val, ks[_])
        if left > right:
            return False
        print(l_val, left, right)
    return True

test_margin = [13,  5, 14,  5,  5, 15,  9,  3,  1,  6,  2,  7,  2,  6,  6,  1]

estimate, err = matrix_count.count_log_symmetric_matrices(test_margin, binary_matrix=True)
print(estimate, err)

print(matrix_count.erdos_gallai_check(test_margin))

print(erdos_gallai_check_parts(test_margin))