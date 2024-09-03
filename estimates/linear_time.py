# Script implementing the linear time estimate of the number of symmetric matrices 
# with given row sums. 

import numpy as np

#################################################
# Dirichlet-multinomial linear time estimates
######################################
# Symmetric matrices
#######################
# Unbounded
def log_Omega_S_DM(row_sums, even_diagonal=False, diagonal_sum=None, alpha=1.0):
    """Dirichlet-multinomial moment-matching estimate of the (log) number of symmetric non-negative matrices with given row sums.

    :param row_sums: Row sums of the matrix.
    :type row_sums: list | np.array
    :param even_diagonal: Whether the diagonal entries of the matrix should be constrained to be even, defaults to False
    :type even_diagonal: bool, optional
    :param diagonal_sum: What the sum of the diagonal elements should be constrained to, a value of None results in no constraint, defaults to None
    :type diagonal_sum: int | None, optional
    :param alpha: Dirichlet-multinomial parameter to weigh the matrices in the sum, a value of 1 gives the uniform count of matrices, defaults to 1
    :type alpha: float, optional
    :return: log(Omega_S), the logarithm of the estimate of the number of symmetric matrices with given row sums
    :rtype: float
    """

    # Validate inputs
    if not isinstance(row_sums, (list, np.ndarray)):
        raise ValueError("The row sums must be provided as a list or numpy array.")
    if not all(isinstance(x, int) for x in row_sums):
        raise ValueError("The row sums must be integers.")
    if not isinstance(even_diagonal, bool):
        raise ValueError("The even_diagonal flag must be a boolean.")
    if diagonal_sum is not None:
        if not isinstance(diagonal_sum, int):
            raise ValueError("The diagonal_sum must be an integer.")
    if not isinstance(alpha, (int, float)):
        raise ValueError("The Dirichlet-multinomial parameter alpha must be a number.")

    # Convert the inputs into numpy arrays
    row_sums = np.array(row_sums)

    N = np.sum(row_sums) # Matrix total
    m = len(row_sums) # Dimension of the (square) matrix

    # If the conditions immediately imply that there are no matrices, return -inf
    if even_diagonal:
        if diagonal_sum is None: # If the diagonal sum is unconstrained, but the diagonal entries must be even the total sum must be even
            if np.sum(row_sums) % 2 != 0:
                return -np.inf
        else:
            if diagonal_sum % 2 != 0: # The sum of the even diagonal entries should be even when it is specified
                return -np.inf
    if diagonal_sum is not None: # If the diagonal sum is specified, we can calculate the sum of the off-diagonal entries
        off_diagonal_sum = np.sum(row_sums) - diagonal_sum
        if off_diagonal_sum % 2 != 0: # The off-diagonal sum must be even since each element is replicated
            return -np.inf
    
    # If these checks are passed, calculate the covariance between two different margin values under the appropriate unrestricted ensemble
    if even_diagonal:
        if diagonal_sum is None:
            # Unrestricted ensemble is over all partitions of half of the total into m_T = m(m+1)/2 parts (upper triangular and diagonal)
            # This is then reflected (and the diagonal is multiplied)
            m_T = m*(m+1)/2
            margin_covariance = (N/2)/m_T*(m_T + N/2)/(m_T + 1)*(m + 2)(-1/m_T)
        else:
            # Unresticted ensemble 

    # We may then convert the covariance into an alpha parameter, 
    # knowing that the general form for the covariance should be (2 N)/m (m alpha + N)/(m alpha + 1)(-m^(-1)).




    return log_Omega_S

#######################
# 0-1 Matrices


######################################
# Asymmetric matrices
#######################
# Unbounded
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

#######################
# 0-1 Matrices


#################################################
# Alternative linear time estimates
######################################
# Symmetric matrices
#######################
# Unbounded

#######################
# 0-1 Matrices

######################################
# Asymmetric matrices
#######################
# Unbounded

#######################
# 0-1 Matrices

