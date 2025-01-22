# Use the sample.pyi implementation of SIS in order to obtain counts and error estimates.

from . import _util
from . import _input_output
import numpy as np
from matrix_count.sample import *


def count_log_symmetric_matrices(row_sums, *, diagonal_sum=None, index_partition=None, block_sums=None, alpha=1.0, estimate_order=3, max_samples=1000, error_target=0.01, verbose=False):
    """Dirichlet-multinomial moment-matching estimate of the logarithm 
        of the number of symmetric non-negative matrices with given row sums.

    :param row_sums: Row sums of the matrix. Length n array-like of non-negative integers. 
    :type row_sums: list | np.array
    :param diagonal_sum: What the sum of the diagonal elements should be constrained to. 
        Either an integer greater than or equal to 0 or None, resulting in no constraint on the diagonal elements, defaults to None.
    :type diagonal_sum: int | None, optional
    :param index_partition: A list of length n of integers ranging from 1 to q. 
        index_partition[i] indicates the block which index i belongs to for the purposes of a block sum constraint. 
        A value of None results in no block sum constraint, defaults to None.
    :type index_partition: list of int | None, optional
    :param block_sums: A 2D (q, q) symmetric square NumPy array of integers representing the constrained sum of each block of the matrix. 
        A value of None results in no block sum constraint, defaults to None.
    :type block_sums: np.ndarray, shape (q, q), dtype int
    :param alpha: Dirichlet-multinomial parameter greater than or equal to 0 to weigh the matrices in the sum.
        A value of 1 gives the uniform count of matrices, defaults to 1
    :type alpha: float, optional
    :param estimate_order: Order of moment matching estimate to use. Options: {2, 3}. Defaults to 3. 
    :type estimate_order: int, optional
    :param max_samples: Maximum number of samples to take. Defaults to 1000.
    :type max_samples: int, optional
    :param error_target: Target absolute error in the logarithm of the count. Defaults to 0.01.
    :type error_target: float, optional
    :param verbose: Whether to print details of calculation. Defaults to False. 
    :type verbose: bool, optional
    :return log_count_est: The logarithm of the number of symmetric matrices under given conditions
    :rtype: float
    :return err: The estimated absolute error in the logarithm of the count
    :rtype: float
    """

    # Check input validity
    _input_output._log_symmetric_matrices_check_arguments(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, estimate_order=estimate_order, verbose=verbose)
    
    # Remove empty margins
    row_sums, diagonal_sum, index_partition, block_sums = _input_output._simplify_input(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums)

    # Check for hardcoded cases
    hardcoded_result = _input_output._log_symmetric_matrices_hardcoded(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, estimate_order=estimate_order, verbose=verbose)
    if hardcoded_result is not None:
        return (hardcoded_result, 0) # No error in the hardcoded cases
    
    # Sample the matrices
    entropies = []
    log_count_est = 0 # Estimated log count
    log_count_err_est = np.inf # Estimated error in the log count
    for i in range(max_samples):
        (sample, entropy) = sample_symmetric_matrix(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, verbose=verbose)
        entropy = entropy + _util._log_weight(sample, alpha) # Really should be averaging w(A)/Q(A), entropy = -log Q(A)
        entropies.append(entropy)
        log_count_est = _util._log_sum_exp(entropies) - np.log(len(entropies))

        logE2 = _util._log_sum_exp(2*np.array(entropies)) - np.log(len(entropies))
        logE = _util._log_sum_exp(entropies) - np.log(len(entropies))
        if logE2 - 2*logE > 0.0001: # Estimate the error by the standard deviation of the counts TODO: treat this better
            log_std = 0.5 * (np.log(np.exp(0) - np.exp(2*logE - logE2)) + logE2)
            log_count_err_est = np.exp(log_std - 0.5 * np.log(len(entropies)) - logE)
            if log_count_err_est < error_target: # Terminate if the error is below the target
                break 
    
    return (log_count_est, log_count_err_est)

