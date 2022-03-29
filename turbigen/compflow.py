"""One-dimensional compressible flow relations.

Use the compflow Python--Fortran package if available (it is faster).
Otherwise use a native implementation."""

from __future__ import absolute_import
import warnings

try:
    from compflow import *
except ImportError:
    from .compflow_native import *

#     warnings.warn(
#         "Falling back to native compflow. This is slower than the Fortan-accelerated compflow package. Try `pip install compflow`."
#     )
