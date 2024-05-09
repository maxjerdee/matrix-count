# Script implementing the linear time estimate of the number of symmetric matrices 
# with given row sums. 

import numpy as np


def log_Omega_S_DM(row_sums, even_diagonal=False, diagonal_sum=None, alpha=1.0):
    """Dirichlet-multinomial moment-matching estimate of the (log) number of symmetric non-negative matrices with given row sums.

    :param row_sums: Row sums of the matrix.
    :type row_sums: list
    :param even_diagonal: Whether the diagonal entries of the matrix should be constrained to be even, defaults to False
    :type even_diagonal: bool, optional
    :param diagonal_sum: What the sum of the diagonal elements should be constrained to, a value of None results in no constraint, defaults to None
    :type diagonal_sum: int | None, optional
    :param alpha: Dirichlet-multinomial parameter to weigh the matrices in the sum, a value of 1 gives the uniform count of matrices, defaults to 1
    :type alpha: float, optional
    :return: log(Omega_S), the logarithm of the estimate of the number of symmetric matrices with given row sums
    :rtype: float
    """

    log_Omega_S = 0

    return log_Omega_S

def log_Omega_DM(row_sums, column_sums, alpha=1.0):
    """Dirichlet-multinomial moment-matching estimate of the (log) number of non-negative matrices with given row and column sums.

    :param row_sums: Row sums of the matrix.
    :type row_sums: list
    :param column_sums: Column sums of the matrix.
    :type column_sums: list
    :param alpha: Dirichlet-multinomial parameter to weigh the matrices in the sum, a value of 1 gives the uniform count of matrices, defaults to 1, defaults to 1
    :type alpha: float, optional
    :return: log(Omega), the logarithm of the estimate of the number of matrices with given row and column sums
    :rtype: float
    """
    
    log_Omega = 0

    return log_Omega

def log_Omega_S_ME(row_sums):
    """Maximum entropy estimate of the (log) number of non-negative symmetric matrices with given row sums.

    :param row_sums: Row sums of the matrix.
    :type row_sums: list
    :return: log(Omega_S), the logarithm of the estimate of the number of symmetric matrices with given row and column sums
    :rtype: float
    """
    log_Omega_S = 0

    return log_Omega_S

def log_Omega_ME(row_sums, column_sums):
    """Maximum entropy estimate of the (log) number of non-negative matrices with given row and column sums.

    :param row_sums: Row sums of the matrix.
    :type row_sums: list
    :param column_sums: Column sums of the matrix.
    :type column_sums: list
    :return: log(Omega), the logarithm of the estimate of the number of matrices with given row and column sums
    :rtype: float
    """
    log_Omega = 0

    return log_Omega
