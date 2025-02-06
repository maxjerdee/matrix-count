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
    # print(ks)
    for l_val in range(1, len(ks) + 1):
        left: int = np.sum(ks[:l_val])
        right: int = l_val * (l_val - 1)
        for _ in range(l_val, len(ks)):
            right += min(l_val, ks[_])
        if left > right:
            return False
        # print(l_val, left, right)
    return True

    # ks = np.array(ks)
    # if np.sum(ks) % 2 == 1 or np.any(ks < 0) or np.any(ks > len(ks) - 1):
    #     return False
    # ks = -np.sort(-ks)
    # for l_val in range(1, len(ks) + 1):
    #     left: int = np.sum(ks[:l_val])
    #     right: int = l_val * (l_val - 1)
    #     for _ in range(l_val, len(ks)):
    #         right += min(l_val, ks[_])
    #     if left > right:
    #         return False
    # return True

test_margin = [2, 16, 14, 30, 9, 24, 11, 25, 12, 4, 12, 8, 11, 11, 24, 1, 8, 10, 13, 16, 20, 9, 13, 12, 8, 19, 22, 7, 19, 1, 6, 3]

estimate, err = matrix_count.count_log_symmetric_matrices(test_margin, binary_matrix=True)
print(estimate, err)

print(matrix_count.erdos_gallai_check(test_margin))

print(erdos_gallai_check_parts(test_margin))