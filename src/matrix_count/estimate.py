from . import _util
from . import _input_output

import numpy as np

######################################
# Symmetric matrices
#######################
# Unbounded 

def alpha_2_symmetric_no_block(matrix_total,n,diagonal_sum=None, alpha=1.0):
    """Dirichlet-Multinomial parameter alpha for the second order moment matching estimate 
        of the number of symmetric matrices with given conditions.

    :param matrix_total: Matrix total (sum of all entries).
    :type matrix_total: int
    :param n: Matrix size (n,n).
    :type n: int
    :param diagonal_sum: Sum of the diagonal elements of the matrix.
    :type diagonal_sum: int | None
    :return: alpha
    :rtype: float
    """
    if diagonal_sum is None:
        # Update this for arbitrary alpha
        numerator = -((1 + n) * (2 + n)) + matrix_total * (2 + n * (2 + n))
        denominator = (-2 + n) * (1 + n) + matrix_total * (2 + n)
        result = numerator / denominator
        return result
    else:
        # Computed for alpha = 1, implemented in Mathematica
        raise NotImplementedError("Not yet implemented.")

def alpha_3_symmetric_no_block(matrix_total,n,diagonal_sum=None, alpha=1.0):
    if diagonal_sum is None:
        common_numerator = (
            8 + matrix_total * (1 + n) * (-16 + n * (-24 + n * (-21 - 4 * n + n**3)))
            - n * (-22 + n * (-23 + n * (-9 + n * (1 + n) * (4 + n))))
            + matrix_total**2 * (8 + n * (18 + n * (22 + n * (13 + n * (5 + n)))))
        )

        sqrt_term = np.sqrt(
            (-1 + matrix_total - n) * n * (matrix_total + n + n**2)
            * (-((1 + n) * (4 + n**2)) + matrix_total * (4 + n * (4 + n * (2 + n))))
            * (-((1 + n) * (4 + n * (5 + n * (6 + n * (3 + n))))) 
            + matrix_total * (4 + n * (9 + n * (10 + n * (8 + n * (3 + n))))))
        )

        denominator = (
            matrix_total * (1 + n) * (-16 + (-2 + n) * n * (2 + n) * (3 + n))
            + (1 + n)**2 * (8 + n * (-2 + n - 3 * n**2 + 2 * n**3))
            + matrix_total**2 * (8 + n * (2 + n) * (7 + n * (2 + n)))
        )

        alpha_plus = (common_numerator + sqrt_term) / denominator
        alpha_minus = (common_numerator - sqrt_term) / denominator

        return alpha_plus, alpha_minus
    else:
        raise NotImplementedError("Not yet implemented.")

# TODO: Calculate and add the block sums cases

def estimate_log_symmetric_matrices(row_sums, *, diagonal_sum=None, index_partition=None, block_sums=None, alpha=1.0, estimate_order=3, verbose=False):
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
    :param verbose: Whether to print details of calculation. Defaults to False. 
    :type verbose: bool, optional
    :return: The logarithm of the estimate of the number of symmetric matrices with given row sums and conditions
    :rtype: float
    """

    # Check input validity
    _input_output._log_symmetric_matrices_check_arguments(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, estimate_order=estimate_order, verbose=verbose)
    
    # Remove empty margins
    row_sums, diagonal_sum, index_partition, block_sums = _input_output._simplify_input(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums)
    
    # Check for hardcoded cases
    hardcoded_result = _input_output._log_symmetric_matrices_hardcoded(row_sums, diagonal_sum=diagonal_sum, index_partition=index_partition, block_sums=block_sums, alpha=alpha, estimate_order=estimate_order)
    if hardcoded_result is not None:
        return hardcoded_result
    else:
        # Actually perform the estimate
        matrix_total = np.sum(row_sums)
        n = len(row_sums)
        if estimate_order == 2:
            alpha_DM = alpha_2_symmetric_no_block(matrix_total,n,diagonal_sum=diagonal_sum, alpha=alpha, estimate_order=estimate_order)
            result = _util._log_binom(matrix_total/2 + n*(n+1)/2 - 1, n*(n+1)/2 - 1) 
            log_P = - _util._log_binom(matrix_total + n*alpha_DM - 1, n*alpha_DM - 1)
            for k in row_sums:
                log_P += _util._log_binom(k + alpha_DM - 1, alpha_DM - 1)
            result += log_P
            return result
        elif estimate_order == 3:
            NotImplementedError("Not yet implemented.") # TODO: complete

#######################
# 0-1 Matrices
# TODO: Implement

######################################
# Asymmetric matrices
#######################
# Unbounded
# TODO: Implement

#######################
# 0-1 Matrices
# TODO: Implement

# ######################################
# # Symmetric matrices
# #######################
# # Unbounded
# def log_Omega_S_DM(row_sums, diagonal_sum=None, alpha=1.0):
#     """Dirichlet-multinomial moment-matching estimate of the (log) number of symmetric non-negative matrices with given row sums.

#     :param row_sums: Row sums of the matrix.
#     :type row_sums: list | np.array
#     :param diagonal_sum: What the sum of the diagonal elements should be constrained to, a value of None results in no constraint, defaults to None
#     :type diagonal_sum: int | None, optional
#     :param alpha: Dirichlet-multinomial parameter to weigh the matrices in the sum, a value of 1 gives the uniform count of matrices, defaults to 1
#     :type alpha: float, optional
#     :return: log(Omega_S), the logarithm of the estimate of the number of symmetric matrices with given row sums
#     :rtype: float
#     """

#     # Validate inputs
#     if not isinstance(row_sums, (list, np.ndarray)):
#         raise ValueError("The row sums must be provided as a list or numpy array.")
#     if not all(isinstance(x, int) for x in row_sums):
#         raise ValueError("The row sums must be integers.")
#     if not isinstance(even_diagonal, bool):
#         raise ValueError("The even_diagonal flag must be a boolean.")
#     if diagonal_sum is not None:
#         if not isinstance(diagonal_sum, int):
#             raise ValueError("The diagonal_sum must be an integer.")
#     if not isinstance(alpha, (int, float)):
#         raise ValueError("The Dirichlet-multinomial parameter alpha must be a number.")

#     # Convert the inputs into numpy arrays
#     row_sums = np.array(row_sums)

#     N = np.sum(row_sums) # Matrix total
#     m = len(row_sums) # Dimension of the (square) matrix

#     # If the conditions immediately imply that there are no matrices, return -inf
#     if even_diagonal:
#         if diagonal_sum is None: # If the diagonal sum is unconstrained, but the diagonal entries must be even the total sum must be even
#             if np.sum(row_sums) % 2 != 0:
#                 return -np.inf
#         else:
#             if diagonal_sum % 2 != 0: # The sum of the even diagonal entries should be even when it is specified
#                 return -np.inf
#     if diagonal_sum is not None: # If the diagonal sum is specified, we can calculate the sum of the off-diagonal entries
#         off_diagonal_sum = np.sum(row_sums) - diagonal_sum
#         if off_diagonal_sum % 2 != 0: # The off-diagonal sum must be even since each element is replicated
#             return -np.inf
    
#     # If these checks are passed, calculate the covariance between two different margin values under the appropriate unrestricted ensemble
#     if even_diagonal:
#         if diagonal_sum is None:
#             # Unrestricted ensemble is over all partitions of half of the total into m_T = m(m+1)/2 parts (upper triangular and diagonal)
#             # This is then reflected (and the diagonal is multiplied)
#             m_T = m*(m+1)/2
#             margin_covariance = (N/2)/m_T*(m_T + N/2)/(m_T + 1)*(m + 2)(-1/m_T)
#         else:
#             # Unresticted ensemble 

#     # We may then convert the covariance into an alpha parameter, 
#     # knowing that the general form for the covariance should be (2 N)/m (m alpha + N)/(m alpha + 1)(-m^(-1)).




#     return log_Omega_S


# def alpha_2(m,n):
#     numerator = -((1 + n) * (2 + n)) + 2 * m * (2 + n * (2 + n))
#     denominator = (-2 + n) * (1 + n) + 2 * m * (2 + n)
#     result = numerator / denominator
#     return result

# # Return the two alphas so that the mixed distribution matches the covariances
# def alpha_3_mixed(m,n):
#     term1 = (1 + n) ** 2 * (128 + n * (160 + n * (224 + n * (136 + n * (79 + 8 * n * (3 + n))))))
#     term2 = 2 * m * (1 + n) * (256 + n * (576 + n * (768 + n * (729 + n * (439 + 16 * n * (14 + n * (4 + n)))))))
#     term3 = 4 * m ** 2 * (128 + n * (416 + n * (672 + n * (753 + 8 * n * (73 + n * (42 + n * (18 + n * (5 + n))))))))
#     sqrt_term = np.sqrt((-1 + 2 * m - n) * n * (2 * m + n + n ** 2) * (term1 - term2 + term3))
    
#     numerator = 32 + 2 * m * (1 + n) * (-64 + n * (-96 + n * (-81 - 19 * n + 4 * n ** 3))) + 4 * m ** 2 * (32 + n * (72 + n * (85 + 4 * n * (13 + n * (5 + n))))) + n * (88 + n * (92 - n * (-39 + n * (10 + n * (17 + 4 * n))))) + sqrt_term
#     numerator_alt = 32 + 2 * m * (1 + n) * (-64 + n * (-96 + n * (-81 - 19 * n + 4 * n ** 3))) + 4 * m ** 2 * (32 + n * (72 + n * (85 + 4 * n * (13 + n * (5 + n))))) + n * (88 + n * (92 - n * (-39 + n * (10 + n * (17 + 4 * n))))) - sqrt_term

#     denominator = 8 * m ** 2 * (4 + n * (2 + n)) * (4 + n * (5 + 2 * n)) + 2 * (1 + n) ** 2 * (16 + (-2 + n) * n * (2 + 3 * n ** 2)) + 4 * m * (1 + n) * (-32 + n * (-24 + (-1 + n) * n * (8 + 3 * n)))

#     alpha1A, alpha1B = numerator / denominator, numerator_alt / denominator

#     term1 = (1 + n) ** 2 * (128 + n * (160 + n * (224 + n * (136 + n * (79 + 8 * n * (3 + n))))))
#     term2 = 2 * m * (1 + n) * (256 + n * (576 + n * (768 + n * (729 + n * (439 + 16 * n * (14 + n * (4 + n)))))))
#     term3 = 4 * m ** 2 * (128 + n * (416 + n * (672 + n * (753 + 8 * n * (73 + n * (42 + n * (18 + n * (5 + n))))))))
#     sqrt_term = np.sqrt((-1 + 2 * m - n) * n * (2 * m + n + n ** 2) * (term1 - term2 + term3))
    
#     numerator = 16 + 2 * m * (1 + n) * (-32 + n * (-48 + n * (-45 - 5 * n + 2 * n ** 3))) + n * (44 + n * (46 - n * (5 + 2 * n) * (-3 + n * (4 + n)))) + 4 * m ** 2 * (16 + n * (36 + n * (47 + 2 * n * (13 + n * (5 + n))))) - sqrt_term
#     numerator_alt = 16 + 2 * m * (1 + n) * (-32 + n * (-48 + n * (-45 - 5 * n + 2 * n ** 3))) + n * (44 + n * (46 - n * (5 + 2 * n) * (-3 + n * (4 + n)))) + 4 * m ** 2 * (16 + n * (36 + n * (47 + 2 * n * (13 + n * (5 + n))))) + sqrt_term

#     denominator = 2 * (1 + n) ** 2 * (8 + n * (-2 + n + 3 * (-1 + n) * n ** 2)) + 16 * m * (-4 + n * (-7 - 4 * n + n ** 3)) + 8 * m ** 2 * (8 + n * (14 + n * (11 + n * (3 + n))))

#     alpha2A, alpha2B = numerator / denominator, numerator_alt / denominator

#     return alpha1A, alpha1B, alpha2A, alpha2B

# # Return the two alphas so that the mixture matches the 
# def alpha_3(m,n):

#     common_numerator = (
#         8 + 2 * m * (1 + n) * (-16 + n * (-24 + n * (-21 - 4 * n + n**3)))
#         - n * (-22 + n * (-23 + n * (-9 + n * (1 + n) * (4 + n))))
#         + 4 * m**2 * (8 + n * (18 + n * (22 + n * (13 + n * (5 + n)))))
#     )

#     sqrt_term = np.sqrt(
#         (-1 + 2 * m - n) * n * (2 * m + n + n**2)
#         * (-((1 + n) * (4 + n**2)) + 2 * m * (4 + n * (4 + n * (2 + n))))
#         * (-((1 + n) * (4 + n * (5 + n * (6 + n * (3 + n))))) 
#            + 2 * m * (4 + n * (9 + n * (10 + n * (8 + n * (3 + n))))))
#     )

#     denominator = (
#         2 * m * (1 + n) * (-16 + (-2 + n) * n * (2 + n) * (3 + n))
#         + (1 + n)**2 * (8 + n * (-2 + n - 3 * n**2 + 2 * n**3))
#         + 4 * m**2 * (8 + n * (2 + n) * (7 + n * (2 + n)))
#     )

#     alpha_plus = (common_numerator + sqrt_term) / denominator
#     alpha_minus = (common_numerator - sqrt_term) / denominator

#     return alpha_plus, alpha_minus

# def log_Omega_2(ks):
#     n = len(ks)
#     m = sum(ks)/2
#     alpha = alpha_2(m,n)
#     result = log_binom(m + n*(n+1)/2 - 1, n*(n+1)/2 - 1) 
#     log_P = - log_binom(2*m + n*alpha - 1, n*alpha - 1)
#     for k in ks:
#         log_P += log_binom(k + alpha - 1, alpha - 1)
#     result += log_P
#     # print(log_P, alpha)
#     # print(ks, result, alpha)
#     return result

# def log_Omega_3(ks):
#     n = len(ks)
#     m = sum(ks)/2
#     alpha_plus, alpha_minus = alpha_3(m,n)
#     result = log_binom(m + n*(n+1)/2 - 1, n*(n+1)/2 - 1) 
#     log_P_plus = - log_binom(2*m + n*alpha_plus - 1, n*alpha_plus - 1)
#     for k in ks:
#         log_P_plus += log_binom(k + alpha_plus - 1, alpha_plus - 1)
#     log_P_minus = - log_binom(2*m + n*alpha_minus - 1, n*alpha_minus - 1)
#     for k in ks:
#         log_P_minus += log_binom(k + alpha_minus - 1, alpha_minus - 1)
#     # print(log_P_plus, log_P_minus, alpha_plus, alpha_minus)
#     result += log_sum_exp([log_P_plus, log_P_minus])-np.log(2)
#     return result


# #######################
# # 0-1 Matrices


# ######################################
# # Asymmetric matrices
# #######################
# # Unbounded
# def log_Omega_DM(row_sums, column_sums, alpha=1.0):
#     """Dirichlet-multinomial moment-matching estimate of the (log) number of non-negative matrices with given row and column sums.

#     :param row_sums: Row sums of the matrix.
#     :type row_sums: list
#     :param column_sums: Column sums of the matrix.
#     :type column_sums: list
#     :param alpha: Dirichlet-multinomial parameter to weigh the matrices in the sum, a value of 1 gives the uniform count of matrices, defaults to 1, defaults to 1
#     :type alpha: float, optional
#     :return: log(Omega), the logarithm of the estimate of the number of matrices with given row and column sums
#     :rtype: float
#     """
    
#     log_Omega = 0

#     return log_Omega

# #######################
# # 0-1 Matrices


# #################################################
# # Alternative linear time estimates
# ######################################
# # Symmetric matrices
# #######################
# # Unbounded

# #######################
# # 0-1 Matrices

# ######################################
# # Asymmetric matrices
# #######################
# # Unbounded

# #######################
# # 0-1 Matrices

