# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Configure pytest for metpy."""

import matplotlib
import numpy
import pint
import scipy


def pytest_report_header(config, startdir):
    """Add dependency information to pytest output."""
    return ('Dependencies: Matplotlib ({}), NumPy ({}), '
            'Pint ({}), SciPy ({})'.format(matplotlib.__version__, numpy.__version__,
                                           pint.__version__, scipy.__version__))
