# Copyright (c) 2016,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Configure pytest for metpy."""

import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pint
import pytest
import scipy

import metpy.calc


def pytest_report_header(config, startdir):
    """Add dependency information to pytest output."""
    return ('Dependencies: Matplotlib ({}), NumPy ({}), Pandas ({}), '
            'Pint ({}), SciPy ({})'.format(matplotlib.__version__, numpy.__version__,
                                           pandas.__version__, pint.__version__,
                                           scipy.__version__))


@pytest.fixture(autouse=True)
def doctest_available_modules(doctest_namespace):
    """Make modules available automatically to doctests."""
    doctest_namespace['metpy'] = metpy
    doctest_namespace['metpy.calc'] = metpy.calc
    doctest_namespace['plt'] = matplotlib.pyplot
