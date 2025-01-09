# Copyright (c) 2014,2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Contains functionality for making meteorological plots."""

from . import cartopy_utils, plot_areas
from ._util import (add_metpy_logo, add_timestamp, add_unidata_logo,  # noqa: F401
                    convert_gempak_color)
from .ctables import *  # noqa: F403
from .declarative import *  # noqa: F403
from .patheffects import *  # noqa: F403
from .skewt import *  # noqa: F403
from .station_plot import *  # noqa: F403
from .text import *  # noqa: F403
from .wx_symbols import *  # noqa: F403
from ..package_tools import set_module

__all__ = ctables.__all__[:]  # pylint: disable=undefined-variable
__all__.extend(declarative.__all__)  # pylint: disable=undefined-variable
__all__.extend(patheffects.__all__)  # pylint: disable=undefined-variable
__all__.extend(skewt.__all__)  # pylint: disable=undefined-variable
__all__.extend(station_plot.__all__)  # pylint: disable=undefined-variable
__all__.extend(text.__all__)  # pylint: disable=undefined-variable
__all__.extend(wx_symbols.__all__)  # pylint: disable=undefined-variable
__all__.extend(['add_metpy_logo', 'add_timestamp', 'add_unidata_logo',
                'convert_gempak_color'])

set_module(globals())


def __getattr__(name):
    """Raise a proper error if something needing Cartopy is not available."""
    for mod in [cartopy_utils, plot_areas]:
        if name in mod.__all__:
            try:
                return getattr(mod, name)
            except AttributeError:
                raise AttributeError(f'Cannot use {name} without Cartopy installed.') from None

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return __all__ + cartopy_utils.__all__ + plot_areas.__all__
