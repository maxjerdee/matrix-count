from __future__ import annotations
from typing import List, Tuple
import numpy as np

def sample_symmetric_matrix_core(ks: List[int], diagonal_sum: int, alpha: float, seed: int) -> Tuple[np.ndarray, float]: ...