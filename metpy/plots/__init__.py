# Copyright (c) 2014,2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Contains functionality for making meteorological plots."""

import logging

# Trigger matplotlib wrappers
from . import _mpl  # noqa: F401
from ._util import add_metpy_logo, add_timestamp, add_unidata_logo, convert_gempak_color
from .ctables import *  # noqa: F403
from .declarative import *  # noqa: F403
from .skewt import *  # noqa: F403
from .station_plot import *  # noqa: F403
from .wx_symbols import *  # noqa: F403

logger = logging.getLogger(__name__)

__all__ = ctables.__all__[:]  # pylint: disable=undefined-variable
__all__.extend(declarative.__all__)  # pylint: disable=undefined-variable
__all__.extend(skewt.__all__)  # pylint: disable=undefined-variable
__all__.extend(station_plot.__all__)  # pylint: disable=undefined-variable
__all__.extend(wx_symbols.__all__)  # pylint: disable=undefined-variable
__all__.extend([add_metpy_logo, add_timestamp, add_unidata_logo,
                convert_gempak_color])  # pylint: disable=undefined-variable
try:
    from .cartopy_utils import USCOUNTIES, USSTATES
    __all__.extend([USCOUNTIES, USSTATES])
except ImportError:
    logger.warning('Cannot import USCOUNTIES without cartopy installed.')
