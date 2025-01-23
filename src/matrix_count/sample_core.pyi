from __future__ import annotations

from numpy.typing import ArrayLike

def sample_symmetric_matrix(
    row_sums: ArrayLike,
    diagonal_sum: int | None = None,
    index_partition: list[int] | None = None,
    block_sums: ArrayLike | None = None,
    alpha: float = 1.0,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[ArrayLike, float]: ...
