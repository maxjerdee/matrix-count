from . import _input_output
from matrix_count.sample_core import sample_symmetric_matrix_core

def sample_symmetric_matrix(row_sums, *, diagonal_sum=None, index_partition=None, block_sums=None, alpha=1.0, estimate_order=3, verbose=False):
    """
    Sample a symmetric matrix with given row sums and diagonal sum.

    :param ks: List of row sums.
    :type ks: list of int
    :param diagonal_sum: Sum of the diagonal elements of the matrix.
    :type diagonal_sum: int
    :return: A symmetric matrix with given row sums and diagonal sum, and the log probability of the matrix.
    :rtype: np.ndarray, float
    """

    # Check input validity
    _input_output._log_symmetric_matrices_check_arguments(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, estimate_order=estimate_order, verbose=verbose)

    # TODO: Implement the block_sums constraint and wrap here.

    diagonal_sum = diagonal_sum if diagonal_sum is not None else -1 # -1 means no constraint

    return sample_symmetric_matrix_core(row_sums, diagonal_sum, alpha)