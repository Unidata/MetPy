# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Subset of constant and thermophysical property values expressed as floats in base units."""

from . import default
from ..units import units

Rd = default.Rd.m_as('m**2 / K / s**2')
Lv = default.Lv.m_as('m**2 / s**2')
Cp_d = default.Cp_d.m_as('m**2 / K / s**2')
zero_degc = units.Quantity(0., 'degC').m_as('K')
sat_pressure_0c = default.sat_pressure_0c.m_as('Pa')
epsilon = default.epsilon.m_as('')
kappa = default.kappa.m_as('')
g = default.g.m_as('m / s**2')
