"""
Copyright (c) 2025 Max Jerdee. All rights reserved.

matrix-count: Counting and sampling non-negative integer matrices given margin sums.
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]

# These are imported in order listed
from matrix_count._util import *
from matrix_count.estimate import *
from matrix_count.sample import *
from matrix_count.count import *
