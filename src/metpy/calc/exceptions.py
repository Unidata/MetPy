# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Contains specific exceptions raised by calculations."""


class InvalidSoundingError(ValueError):
    """Raise when a sounding does not meet thermodynamic calculation expectations."""
