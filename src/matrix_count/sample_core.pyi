from __future__ import annotations

import numpy as np

def sample_symmetric_matrix_core(
    ks: list[int], diagonal_sum: int, alpha: float, seed: int
) -> tuple[np.ndarray, float]: ...
