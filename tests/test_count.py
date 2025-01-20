from __future__ import annotations

from matrix_count.count import *
import numpy as np
from pytest import approx, raises


def test_count_log_symmetric_matrices():
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3])
    log_count_true = np.log(34)
    assert log_count_est == approx(log_count_true, abs=3*log_count_est_err) # Check within 3 sigma, chance that this just randomly fails
    
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3], alpha=5)
    log_count_true = np.log(693809375)
    assert log_count_est == approx(log_count_true, abs=3*log_count_est_err) # Check within 3 sigma, chance that this just randomly fails

    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3], diagonal_sum=20)
    log_count_true = 1.09861
    assert log_count_est == approx(log_count_true, abs=3*log_count_est_err) # Check within 3 sigma, chance that this just randomly fails
    
    (log_count_est, log_count_est_err) = count_log_symmetric_matrices([20,11,3], diagonal_sum=20, alpha=5)
    log_count_true = 18.4677
    assert log_count_est == approx(log_count_true, abs=3*log_count_est_err) # Check within 3 sigma, chance that this just randomly fails