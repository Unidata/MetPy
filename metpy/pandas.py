# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Provide accessors to enhance interoperability between XArray and MetPy."""
from __future__ import absolute_import
from collections.abc import Iterable

import functools
import logging

import numpy as np
import pandas as pd

__all__ = []
log = logging.getLogger(__name__)


def preprocess_pandas(func):
    """Decorate a function to convert all data series arguments to np.ndarray.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # not using hasattr(a, values) because it picks up dict.values()
        # and this is more explictly handling pandas
        args = tuple(a.values if isinstance(a, pd.Series) else a for a in args)
        kwargs = {name: (v.values if isinstance(v, pd.Series) else v)
                  for name, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper
