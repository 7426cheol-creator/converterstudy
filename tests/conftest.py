"""Test session setup: pin BLAS to one thread before numpy is imported (see convlab/__init__.py)."""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
