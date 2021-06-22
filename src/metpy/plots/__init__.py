# Copyright (c) 2014,2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Contains functionality for making meteorological plots."""

# Trigger matplotlib wrappers
from . import _mpl  # noqa: F401
from . import cartopy_utils
from ._util import (add_metpy_logo, add_timestamp, add_unidata_logo,  # noqa: F401
                    convert_gempak_color)
from .ctables import *  # noqa: F403
from .declarative import *  # noqa: F403
from .skewt import *  # noqa: F403
from .station_plot import *  # noqa: F403
from .wx_symbols import *  # noqa: F403
from ..package_tools import set_module

__all__ = ctables.__all__[:]  # pylint: disable=undefined-variable
__all__.extend(declarative.__all__)  # pylint: disable=undefined-variable
__all__.extend(skewt.__all__)  # pylint: disable=undefined-variable
__all__.extend(station_plot.__all__)  # pylint: disable=undefined-variable
__all__.extend(wx_symbols.__all__)  # pylint: disable=undefined-variable
__all__.extend(['add_metpy_logo', 'add_timestamp', 'add_unidata_logo',
                'convert_gempak_color'])

set_module(globals())


def __getattr__(name):
    """Handle warning if Cartopy map features are not available."""
    if name in cartopy_utils.__all__:
        try:
            return getattr(cartopy_utils, name)
        except AttributeError:
            raise AttributeError(f'Cannot use {name} without Cartopy installed.') from None

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return __all__ + cartopy_utils.__all__
