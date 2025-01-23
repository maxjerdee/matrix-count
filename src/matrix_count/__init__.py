"""
Copyright (c) 2025 Max Jerdee. All rights reserved.

matrix-count: Counting and sampling non-negative integer matrices given margin sums.
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = [
    "__version__",
    "_log_binom",
    "_log_factorial",
    "_log_factorial2",
    "_log_sum_exp",
    "count_log_symmetric_matrices",
    "estimate_log_symmetric_matrices",
    "sample_symmetric_matrix",
]

# These are imported in order listed
from matrix_count._util import _log_binom, _log_factorial, _log_factorial2, _log_sum_exp
from matrix_count.count import count_log_symmetric_matrices
from matrix_count.estimate import estimate_log_symmetric_matrices
from matrix_count.sample import sample_symmetric_matrix
