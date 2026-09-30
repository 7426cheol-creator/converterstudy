"""Converter FAE Lab - textbook v4.0-aligned learning simulator (FL01-FL12, EX01-EX12)."""

import os as _os

# The engine multiplies and exponentiates many tiny (<= 10 x 10) matrices.  A multi-threaded BLAS
# gains nothing there, and when several processes or server threads share the cores its spinning
# threads make each call orders of magnitude slower (measured: 21 ms instead of 40 us per 5 x 5
# expm).  Pin BLAS to one thread unless the user has chosen otherwise; this must run before numpy
# is imported, which is why it lives in the package's __init__.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

__version__ = "4.0.0"
CONTRACT_VERSION = "4.0-reconstructed"  # original simulation_contract_v4.json was not provided; see docs/ERRATA.md
