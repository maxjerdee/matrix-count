from __future__ import annotations

from matrix_count.count import *
import numpy as np
from pytest import approx, raises

def test_count_log_symmetric_matrices_no_constraints():
    sigma_error = 4 # Number of standard deviations to check within

    # 20,11,3
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3])
    log_count_true = np.log(34)
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails
    
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3], alpha=5)
    log_count_true = np.log(693809375)
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails

    # 3,3,3,3,2,2
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([3,3,3,3,2,2])
    log_count_true = 7.51098
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails
    
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([3,3,3,3,2,2], alpha=5)
    log_count_true = 19.8828
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails
    

def test_count_log_symmetric_matrices_diagonal_sum():
    sigma_error = 4 # Number of standard deviations to check within
    
    # 20,11,3
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3], diagonal_sum=20)
    log_count_true = 1.09861
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails
    
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3], diagonal_sum=20, alpha=5)
    log_count_true = 18.4677
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails

    # 3,3,3,3,2,2
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([3,3,3,3,2,2], diagonal_sum=10)
    log_count_true = 2.77259
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails
    
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([3,3,3,3,2,2], alpha=5, diagonal_sum=10)
    log_count_true = 15.6481
    assert log_count_est == approx(log_count_true, abs=sigma_error*log_count_est_err) # Chance that this just randomly fails
    