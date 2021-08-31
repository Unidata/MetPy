# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the solver."""
import pytest

from metpy.calc.field_solver import Solver

test_solver = Solver()

@test_solver.register('Td')
def dewpoint_from_relative_humidity(temperature, relative_humidity):
    pass


@test_solver.register('RH')
def relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity):
    pass


@test_solver.register('r')
def mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity):
    pass


@test_solver.register('Tv')
def virtual_temperature(temperature, mixing_ratio, molecular_weight_ratio=5):
    pass


@test_solver.register('RH')
def relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio):
    pass


@test_solver.register('r')
def mixing_ratio_from_specific_humidity(specific_humidity):
    pass


@test_solver.register('q')
def specific_humidity_from_mixing_ratio(mixing_ratio):
    pass


@test_solver.register('rho')
def density(pressure, temperature, mixing_ratio):
    pass


@test_solver.register('Tw')
def wet_bulb_temperature(pressure, temperature, dewpoint):
    pass


@test_solver.register('u', 'v')
def wind_components(wind_speed, wind_direction):
    pass

@test_solver.register()
def vorticity(u, v):
    pass

@pytest.mark.parametrize(['inputs', 'want', 'truth'], [
    ({'T', 'RH'}, 'Td', [dewpoint_from_relative_humidity]),
    ({'T', 'p', 'q'}, 'Td', [relative_humidity_from_specific_humidity,
                             dewpoint_from_relative_humidity]),
    ({'T', 'p', 'q'}, 'Tv', [mixing_ratio_from_specific_humidity, virtual_temperature]),
    ({'p', 'T', 'RH'}, 'rho', [mixing_ratio_from_relative_humidity, density]),
    ({'p', 'T', 'RH'}, 'Tw', [dewpoint_from_relative_humidity, wet_bulb_temperature]),
    ({'wind_speed', 'wind_direction'}, 'vorticity', [wind_components, vorticity])
])
def test_solutions(inputs, want, truth):
    """Test that the proper sequence of calculations is found."""
    assert test_solver.solve(inputs, want) == truth


def test_failure():
    """Test that the correct error results when a value cannot be solved."""
    with pytest.raises(ValueError):
        test_solver.solve({'RH'}, 'Td')
