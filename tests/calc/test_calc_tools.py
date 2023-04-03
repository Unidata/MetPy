# Copyright (c) 2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the `tools` module."""

from collections import namedtuple

import numpy as np
import numpy.ma as ma
import pandas as pd
from pyproj import CRS, Geod
import pytest
import xarray as xr

from metpy.calc import (angle_to_direction, find_bounding_indices, find_intersections,
                        first_derivative, geospatial_gradient, get_layer, get_layer_heights,
                        gradient, laplacian, lat_lon_grid_deltas, nearest_intersection_idx,
                        parse_angle, pressure_to_height_std, reduce_point_density,
                        resample_nn_1d, second_derivative, vector_derivative)
from metpy.calc.tools import (_delete_masked_points, _get_bound_pressure_height,
                              _greater_or_close, _less_or_close, _next_non_masked_element,
                              _remove_nans, azimuth_range_to_lat_lon, BASE_DEGREE_MULTIPLIER,
                              DIR_STRS, nominal_lat_lon_grid_deltas, parse_grid_arguments, UND)
from metpy.testing import (assert_almost_equal, assert_array_almost_equal, assert_array_equal,
                           get_test_data)
from metpy.units import units
from metpy.xarray import grid_deltas_from_dataarray, preprocess_and_wrap

FULL_CIRCLE_DEGREES = np.arange(0, 360, BASE_DEGREE_MULTIPLIER.m) * units.degree


def test_resample_nn():
    """Test 1d nearest neighbor functionality."""
    a = np.arange(5.)
    b = np.array([2, 3.8])
    truth = np.array([2, 4])

    assert_array_equal(truth, resample_nn_1d(a, b))


def test_nearest_intersection_idx():
    """Test nearest index to intersection functionality."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x**2
    y2 = 100 * x - 650
    truth = np.array([2, 12])

    assert_array_equal(truth, nearest_intersection_idx(y1, y2))


@pytest.mark.parametrize('direction, expected', [
    ('all', np.array([[8.88, 24.44], [238.84, 1794.53]])),
    ('increasing', np.array([[24.44], [1794.53]])),
    ('decreasing', np.array([[8.88], [238.84]]))
])
def test_find_intersections(direction, expected):
    """Test finding the intersection of two curves functionality."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x**2
    y2 = 100 * x - 650
    # Note: Truth is what we will get with this sampling, not the mathematical intersection
    assert_array_almost_equal(expected, find_intersections(x, y1, y2, direction=direction), 2)


def test_find_intersections_no_intersections():
    """Test finding the intersection of two curves with no intersections."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x + 0
    y2 = 5 * x + 5
    # Note: Truth is what we will get with this sampling, not the mathematical intersection
    truth = np.array([[],
                      []])
    assert_array_equal(truth, find_intersections(x, y1, y2))


def test_find_intersections_invalid_direction():
    """Test exception if an invalid direction is given."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x ** 2
    y2 = 100 * x - 650
    with pytest.raises(ValueError):
        find_intersections(x, y1, y2, direction='increaing')


def test_find_intersections_units():
    """Test handling of units when logarithmic interpolation is called."""
    x = np.linspace(5, 30, 17) * units.hPa
    y1 = 3 * x.m**2
    y2 = 100 * x.m - 650
    truth = np.array([24.43, 1794.54])
    x_test, y_test = find_intersections(x, y1, y2, direction='increasing', log_x=True)
    assert_array_almost_equal(truth, np.array([x_test.m, y_test.m]).flatten(), 2)
    assert x_test.units == units.hPa


@pytest.mark.parametrize('direction, expected', [
    ('all', np.array([[0., 3.5, 4.33333333, 7., 9., 10., 11.5, 13.], np.zeros(8)])),
    ('increasing', np.array([[0., 4.333, 7., 11.5], np.zeros(4)])),
    ('decreasing', np.array([[3.5, 10.], np.zeros(2)]))
])
def test_find_intersections_intersections_in_data_at_ends(direction, expected):
    """Test finding intersections when intersections are in the data.

    Test data includes points of intersection, sequential points of intersection, intersection
    at the ends of the data, and intersections in increasing/decreasing direction.
    """
    x = np.arange(14)
    y1 = np.array([0, 3, 2, 1, -1, 2, 2, 0, 1, 0, 0, -2, 2, 0])
    y2 = np.zeros_like(y1)
    assert_array_almost_equal(expected, find_intersections(x, y1, y2, direction=direction), 2)


@pytest.mark.parametrize('mask, expected_idx, expected_element', [
    ([False, False, False, False, False], 1, 1),
    ([False, True, True, False, False], 3, 3),
    ([False, True, True, True, True], None, None)
])
def test_non_masked_elements(mask, expected_idx, expected_element):
    """Test with a valid element."""
    a = ma.masked_array(np.arange(5), mask=mask)
    idx, element = _next_non_masked_element(a, 1)
    assert idx == expected_idx
    assert element == expected_element


@pytest.fixture
def thin_point_data():
    r"""Provide scattered points for testing."""
    return np.array([[0.8793620, 0.9005706], [0.5382446, 0.8766988], [0.6361267, 0.1198620],
                     [0.4127191, 0.0270573], [0.1486231, 0.3121822], [0.2607670, 0.4886657],
                     [0.7132257, 0.2827587], [0.4371954, 0.5660840], [0.1318544, 0.6468250],
                     [0.6230519, 0.0682618], [0.5069460, 0.2326285], [0.1324301, 0.5609478],
                     [0.7975495, 0.2109974], [0.7513574, 0.9870045], [0.9305814, 0.0685815],
                     [0.5271641, 0.7276889], [0.8116574, 0.4795037], [0.7017868, 0.5875983],
                     [0.5591604, 0.5579290], [0.1284860, 0.0968003], [0.2857064, 0.3862123]])


@pytest.mark.parametrize('radius, truth',
                         [(2.0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)),
                          (1.0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)),
                          (0.3, np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
                                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)),
                          (0.1, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                                          0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool))
                          ])
def test_reduce_point_density(thin_point_data, radius, truth):
    r"""Test that reduce_point_density works."""
    assert_array_equal(reduce_point_density(thin_point_data, radius=radius), truth)


def test_reduce_point_density_nonfinite():
    """Test that non-finite point values are properly marked not to keep."""
    points = np.array(
        [(np.nan, np.nan), (np.nan, 5), (5, 5), (5, np.nan),
         (10, 10), (np.inf, 10), (10, np.inf), (np.inf, np.inf)])
    mask = reduce_point_density(points, 1)
    assert_array_equal(mask, [False, False, True, False, True, False, False, False])


@pytest.mark.parametrize('radius, truth',
                         [(2.0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)),
                          (1.0, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)),
                          (0.3, np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
                                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)),
                          (0.1, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                                          0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool))
                          ])
def test_reduce_point_density_units(thin_point_data, radius, truth):
    r"""Test that reduce_point_density works with units."""
    assert_array_equal(reduce_point_density(thin_point_data * units.dam,
                                            radius=radius * units.dam), truth)


@pytest.mark.parametrize('radius, truth',
                         [(2.0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)),
                          (0.7, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=bool)),
                          (0.3, np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
                                          0, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=bool)),
                          (0.1, np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool))
                          ])
def test_reduce_point_density_priority(thin_point_data, radius, truth):
    r"""Test that reduce_point_density works properly with priority."""
    key = np.array([8, 6, 2, 8, 6, 4, 4, 8, 8, 6, 3, 4, 3, 0, 7, 4, 3, 2, 3, 3, 9])
    assert_array_equal(reduce_point_density(thin_point_data, radius, key), truth)


def test_reduce_point_density_1d():
    r"""Test that reduce_point_density works with 1D points."""
    x = np.array([1, 3, 4, 8, 9, 10])
    assert_array_equal(reduce_point_density(x, 2.5),
                       np.array([1, 0, 1, 1, 0, 0], dtype=bool))


def test_delete_masked_points():
    """Test deleting masked points."""
    a = ma.masked_array(np.arange(5), mask=[False, True, False, False, False])
    b = ma.masked_array(np.arange(5), mask=[False, False, False, True, False])
    expected = np.array([0, 2, 4])
    a, b = _delete_masked_points(a, b)
    assert_array_equal(a, expected)
    assert_array_equal(b, expected)


def get_bounds_data():
    """Provide pressure and height data for testing layer bounds calculation."""
    pressures = np.linspace(1000, 100, 10) * units.hPa
    heights = pressure_to_height_std(pressures)
    return pressures, heights


@pytest.mark.parametrize('pressure, bound, hgts, interp, expected', [
    (get_bounds_data()[0], 900 * units.hPa, None, True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 900 * units.hPa, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, None, True,
     (870 * units.hPa, 1.2665298 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, None, True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, None, True,
     (870 * units.hPa, 1.2665298 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, None, False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 900 * units.hPa, get_bounds_data()[1], True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 900 * units.hPa, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, get_bounds_data()[1], True,
     (870 * units.hPa, 1.2643214 * units.kilometer)),
    (get_bounds_data()[0], 870 * units.hPa, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, get_bounds_data()[1], True,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.9880028 * units.kilometer, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, get_bounds_data()[1], True,
     (870.9869087 * units.hPa, 1.2665298 * units.kilometer)),
    (get_bounds_data()[0], 1.2665298 * units.kilometer, get_bounds_data()[1], False,
     (900 * units.hPa, 0.9880028 * units.kilometer)),
    (get_bounds_data()[0], 0.98800289 * units.kilometer, get_bounds_data()[1], True,
     (900 * units.hPa, 0.9880028 * units.kilometer))
])
def test_get_bound_pressure_height(pressure, bound, hgts, interp, expected):
    """Test getting bounds in layers with various parameter combinations."""
    bounds = _get_bound_pressure_height(pressure, bound, height=hgts, interpolate=interp)
    assert_array_almost_equal(bounds[0], expected[0], 2)
    assert_array_almost_equal(bounds[1], expected[1], 4)


def test_get_bound_invalid_bound_units():
    """Test that value error is raised with invalid bound units."""
    p = np.arange(900, 300, -100) * units.hPa
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 100 * units.degC)


def test_get_bound_pressure_out_of_range():
    """Test when bound is out of data range in pressure."""
    p = np.arange(900, 300, -100) * units.hPa
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 100 * units.hPa)
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 1000 * units.hPa)


def test_get_bound_height_out_of_range():
    """Test when bound is out of data range in height."""
    p = np.arange(900, 300, -100) * units.hPa
    h = np.arange(1, 7) * units.kilometer
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 8 * units.kilometer, height=h)
    with pytest.raises(ValueError):
        _get_bound_pressure_height(p, 100 * units.meter, height=h)


@pytest.mark.parametrize('flip_order', [(True, False)])
def test_get_layer_float32(flip_order):
    """Test that get_layer works properly with float32 data."""
    p = np.asarray([940.85083008, 923.78851318, 911.42022705, 896.07220459,
                    876.89404297, 781.63330078], np.float32) * units('hPa')
    hgt = np.asarray([563.671875, 700.93817139, 806.88098145, 938.51745605,
                      1105.25854492, 2075.04443359], dtype=np.float32) * units.meter

    true_p_layer = np.asarray([940.85083008, 923.78851318, 911.42022705, 896.07220459,
                               876.89404297, 831.86472819], np.float32) * units('hPa')
    true_hgt_layer = np.asarray([563.671875, 700.93817139, 806.88098145, 938.51745605,
                                 1105.25854492, 1549.8079], dtype=np.float32) * units.meter

    if flip_order:
        p = p[::-1]
        hgt = hgt[::-1]
    p_layer, hgt_layer = get_layer(p, hgt, height=hgt, depth=1000. * units.meter)
    assert_array_almost_equal(p_layer, true_p_layer, 4)
    assert_array_almost_equal(hgt_layer, true_hgt_layer, 4)


def test_get_layer_float32_no_heights():
    """Test that get_layer works with float32 data when not given heights."""
    p = np.array([1017.695312, 1010.831787, 1002.137207, 991.189453, 977.536194, 960.655212,
                  940.116455, 915.509521, 886.550415], dtype=np.float32) * units.hPa
    u = np.array([0.205419, 0.206133, -0.354010, -1.586414, -2.660765, -3.740533,
                  -3.297433, 1.049471, 5.610486], dtype=np.float32) * units('m/s')
    v = np.array([6.491890, 8.920976, 13.959625, 18.398054, 21.416298, 23.190233,
                  23.028181, 20.971205, 19.243179], dtype=np.float32) * units('m/s')

    p_l, u_l, v_l = get_layer(p, u, v, depth=1000 * units.meter)
    assert_array_equal(p_l[:-1], p[:-1])
    assert_array_almost_equal(u_l[:-1], u[:-1], 7)
    assert_almost_equal(u_l[-1], 3.0455916 * units('m/s'), 4)
    assert_array_almost_equal(v_l[:-1], v[:-1], 7)
    assert_almost_equal(v_l[-1], 20.2149378 * units('m/s'), 4)
    assert p_l.dtype == p.dtype
    assert u_l.dtype == u.dtype
    assert v_l.dtype == v.dtype


def test_get_layer_ragged_data():
    """Test that an error is raised for unequal length pressure and data arrays."""
    p = np.arange(10) * units.hPa
    y = np.arange(9) * units.degC
    with pytest.raises(ValueError):
        get_layer(p, y)


def test_get_layer_invalid_depth_units():
    """Test that an error is raised when depth has invalid units."""
    p = np.arange(10) * units.hPa
    y = np.arange(9) * units.degC
    with pytest.raises(ValueError):
        get_layer(p, y, depth=400 * units.degC)


def layer_test_data():
    """Provide test data for testing of layer bounds."""
    pressure = np.arange(1000, 10, -100) * units.hPa
    temperature = np.linspace(25, -50, len(pressure)) * units.degC
    return pressure, temperature


@pytest.mark.parametrize('pressure, variable, heights, bottom, depth, interp, expected', [
    (layer_test_data()[0], layer_test_data()[1], None, None, 150 * units.hPa, True,
     (np.array([1000, 900, 850]) * units.hPa,
      np.array([25.0, 16.666666, 12.62262]) * units.degC)),
    (layer_test_data()[0], layer_test_data()[1], None, None, 150 * units.hPa, False,
     (np.array([1000, 900]) * units.hPa, np.array([25.0, 16.666666]) * units.degC)),
    (layer_test_data()[0], layer_test_data()[1], None, 2 * units.km, 3 * units.km, True,
     (np.array([794.85264282, 700., 600., 540.01696548]) * units.hPa,
      np.array([7.93049516, 0., -8.33333333, -13.14758845]) * units.degC))
])
def test_get_layer(pressure, variable, heights, bottom, depth, interp, expected):
    """Test get_layer functionality."""
    p_layer, y_layer = get_layer(pressure, variable, height=heights, bottom=bottom,
                                 depth=depth, interpolate=interp)
    assert_array_almost_equal(p_layer, expected[0], 2)
    assert_array_almost_equal(y_layer, expected[1], 3)


def test_get_layer_units():
    """Test get_layer when height profile has different units from bottom and depth."""
    pressure, temperature = layer_test_data()
    height = units.Quantity(np.linspace(100, 50000, len(pressure)), 'm')
    pres_subset, temp_subset = get_layer(pressure, temperature,
                                         bottom=units.Quantity(1, 'km'),
                                         depth=units.Quantity(5, 'km'),
                                         height=height)
    pres_expected = units.Quantity([983, 900, 893], 'hPa')
    temp_expected = units.Quantity([23.64385006, 16.66666667, 16.11422559], 'degC')
    assert_array_almost_equal(pres_subset, pres_expected, 2)
    assert_array_almost_equal(temp_subset, temp_expected, 3)


def test_get_layer_masked():
    """Test get_layer with masked arrays as input."""
    p = units.Quantity(np.ma.array([1000, 500, 400]), 'hPa')
    u = units.Quantity(np.arange(3), 'm/s')
    p_layer, u_layer = get_layer(p, u, depth=units.Quantity(6000, 'm'))
    true_p_layer = units.Quantity([1000., 500., 464.4742], 'hPa')
    true_u_layer = units.Quantity([0., 1., 1.3303], 'm/s')
    assert_array_almost_equal(p_layer, true_p_layer, 4)
    assert_array_almost_equal(u_layer, true_u_layer, 4)


def test_greater_or_close():
    """Test floating point greater or close to."""
    x = np.array([0.0, 1.0, 1.49999, 1.5, 1.5000, 1.7])
    comparison_value = 1.5
    truth = np.array([False, False, True, True, True, True])
    res = _greater_or_close(x, comparison_value)
    assert_array_equal(res, truth)


def test_greater_or_close_mixed_types():
    """Test _greater_or_close with mixed Quantity and array errors."""
    with pytest.raises(ValueError):
        _greater_or_close(1000. * units.mbar, 1000.)

    with pytest.raises(ValueError):
        _greater_or_close(1000., 1000. * units.mbar)


def test_less_or_close():
    """Test floating point less or close to."""
    x = np.array([0.0, 1.0, 1.49999, 1.5, 1.5000, 1.7])
    comparison_value = 1.5
    truth = np.array([True, True, True, True, True, False])
    res = _less_or_close(x, comparison_value)
    assert_array_equal(res, truth)


def test_less_or_close_mixed_types():
    """Test _less_or_close with mixed Quantity and array errors."""
    with pytest.raises(ValueError):
        _less_or_close(1000. * units.mbar, 1000.)

    with pytest.raises(ValueError):
        _less_or_close(1000., 1000. * units.mbar)


def test_get_layer_heights_interpolation():
    """Test get_layer_heights with interpolation."""
    heights = np.arange(10) * units.km
    data = heights.m * 2 * units.degC
    heights, data = get_layer_heights(heights, 5000 * units.m, data, bottom=1500 * units.m)
    heights_true = np.array([1.5, 2, 3, 4, 5, 6, 6.5]) * units.km
    data_true = heights_true.m * 2 * units.degC
    assert_array_almost_equal(heights_true, heights, 6)
    assert_array_almost_equal(data_true, data, 6)


def test_get_layer_heights_no_interpolation():
    """Test get_layer_heights without interpolation."""
    heights = np.arange(10) * units.km
    data = heights.m * 2 * units.degC
    heights, data = get_layer_heights(heights, 5000 * units.m, data,
                                      bottom=1500 * units.m, interpolate=False)
    heights_true = np.array([2, 3, 4, 5, 6]) * units.km
    data_true = heights_true.m * 2 * units.degC
    assert_array_almost_equal(heights_true, heights, 6)
    assert_array_almost_equal(data_true, data, 6)


def test_get_layer_heights_agl():
    """Test get_layer_heights with interpolation."""
    heights = np.arange(300, 1200, 100) * units.m
    data = heights.m * 0.1 * units.degC
    heights, data = get_layer_heights(heights, 500 * units.m, data, with_agl=True)
    heights_true = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]) * units.km
    data_true = np.array([30, 40, 50, 60, 70, 80]) * units.degC
    assert_array_almost_equal(heights_true, heights, 6)
    assert_array_almost_equal(data_true, data, 6)


def test_get_layer_heights_agl_bottom_no_interp():
    """Test get_layer_heights with no interpolation and a bottom."""
    heights_init = np.arange(300, 1200, 100) * units.m
    data = heights_init.m * 0.1 * units.degC
    heights, data = get_layer_heights(heights_init, 500 * units.m, data, with_agl=True,
                                      interpolate=False, bottom=200 * units.m)
    # Regression test for #789
    assert_array_equal(heights_init[0], 300 * units.m)
    heights_true = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) * units.km
    data_true = np.array([50, 60, 70, 80, 90, 100]) * units.degC
    assert_array_almost_equal(heights_true, heights, 6)
    assert_array_almost_equal(data_true, data, 6)


def test_lat_lon_grid_deltas_1d():
    """Test for lat_lon_grid_deltas for variable grid."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    dx, dy = lat_lon_grid_deltas(lon, lat)
    dx_truth = np.array([[212943.5585, 212943.5585, 212943.5585],
                         [204946.2305, 204946.2305, 204946.2305],
                         [196558.8269, 196558.8269, 196558.8269],
                         [187797.3216, 187797.3216, 187797.3216]]) * units.meter
    dy_truth = np.array([[277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857]]) * units.meter
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


@pytest.mark.parametrize('flip_order', [(False, True)])
def test_lat_lon_grid_deltas_2d(flip_order):
    """Test for lat_lon_grid_deltas for variable grid with negative delta distances."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    dx_truth = np.array([[212943.5585, 212943.5585, 212943.5585],
                         [204946.2305, 204946.2305, 204946.2305],
                         [196558.8269, 196558.8269, 196558.8269],
                         [187797.3216, 187797.3216, 187797.3216]]) * units.meter
    dy_truth = np.array([[277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857],
                         [277987.1857, 277987.1857, 277987.1857, 277987.1857]]) * units.meter
    if flip_order:
        lon = lon[::-1]
        lat = lat[::-1]
        dx_truth = -1 * dx_truth[::-1]
        dy_truth = -1 * dy_truth[::-1]

    lon, lat = np.meshgrid(lon, lat)
    dx, dy = lat_lon_grid_deltas(lon, lat)
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


def test_lat_lon_grid_deltas_extra_dimensions():
    """Test for lat_lon_grid_deltas with extra leading dimensions."""
    lon, lat = np.meshgrid(np.arange(-100, -90, 2.5), np.arange(40, 50, 2.5))
    lat = lat[None, None]
    lon = lon[None, None]
    dx_truth = np.array([[[[212943.5585, 212943.5585, 212943.5585],
                           [204946.2305, 204946.2305, 204946.2305],
                           [196558.8269, 196558.8269, 196558.8269],
                           [187797.3216, 187797.3216, 187797.3216]]]]) * units.meter
    dy_truth = (np.array([[[[277987.1857, 277987.1857, 277987.1857, 277987.1857],
                            [277987.1857, 277987.1857, 277987.1857, 277987.1857],
                            [277987.1857, 277987.1857, 277987.1857, 277987.1857]]]])
                * units.meter)
    dx, dy = lat_lon_grid_deltas(lon, lat)
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


def test_lat_lon_grid_deltas_mismatched_shape():
    """Test for lat_lon_grid_deltas for variable grid."""
    lat = np.arange(40, 50, 2.5)
    lon = np.array([[-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5],
                    [-100., -97.5, -95., -92.5]])
    with pytest.raises(ValueError):
        lat_lon_grid_deltas(lon, lat)


def test_lat_lon_grid_deltas_geod_kwargs():
    """Test that geod kwargs are overridden by users #774."""
    lat = np.arange(40, 50, 2.5)
    lon = np.arange(-100, -90, 2.5)
    dx, dy = lat_lon_grid_deltas(lon, lat, geod=Geod(a=4370997))
    dx_truth = np.array([[146095.76101984, 146095.76101984, 146095.76101984],
                         [140608.9751528, 140608.9751528, 140608.9751528],
                         [134854.56713287, 134854.56713287, 134854.56713287],
                         [128843.49645823, 128843.49645823, 128843.49645823]]) * units.meter
    dy_truth = np.array([[190720.72311199, 190720.72311199, 190720.72311199, 190720.72311199],
                         [190720.72311199, 190720.72311199, 190720.72311199, 190720.72311199],
                         [190720.72311199, 190720.72311199, 190720.72311199,
                          190720.72311199]]) * units.meter
    assert_almost_equal(dx, dx_truth, 4)
    assert_almost_equal(dy, dy_truth, 4)


@pytest.fixture()
def deriv_1d_data():
    """Return 1-dimensional data for testing derivative functions."""
    return namedtuple('D_1D_Test_Data', 'x values')(np.array([0, 1.25, 3.75]) * units.cm,
                                                    np.array([13.5, 12, 10]) * units.degC)


@pytest.fixture()
def deriv_2d_data():
    """Return 2-dimensional data for analytic function for testing derivative functions."""
    ret = namedtuple('D_2D_Test_Data', 'x y x0 y0 a b f')(
        np.array([0., 2., 7.]), np.array([1., 5., 11., 13.]), 3, 1.5, 0.5, 0.25, 0)

    # Makes a value array with y changing along rows (axis 0) and x along columns (axis 1)
    return ret._replace(f=ret.a * (ret.x - ret.x0)**2 + ret.b * (ret.y[:, None] - ret.y0)**2)


@pytest.fixture()
def deriv_4d_data():
    """Return simple 4-dimensional data for testing axis handling of derivative functions."""
    return np.arange(3 * 3 * 4 * 4).reshape((3, 3, 4, 4))


def test_first_derivative(deriv_1d_data):
    """Test first_derivative with a simple 1D array."""
    dv_dx = first_derivative(deriv_1d_data.values, x=deriv_1d_data.x)

    # Worked by hand and taken from Chapra and Canale 23.2
    truth = np.array([-1.333333, -1.06666667, -0.5333333]) * units('delta_degC / cm')
    assert_array_almost_equal(dv_dx, truth, 5)


def test_first_derivative_2d(deriv_2d_data):
    """Test first_derivative with a full 2D array."""
    df_dx = first_derivative(deriv_2d_data.f, x=deriv_2d_data.x, axis=1)
    df_dx_analytic = np.tile(2 * deriv_2d_data.a * (deriv_2d_data.x - deriv_2d_data.x0),
                             (deriv_2d_data.f.shape[0], 1))
    assert_array_almost_equal(df_dx, df_dx_analytic, 5)

    df_dy = first_derivative(deriv_2d_data.f, x=deriv_2d_data.y, axis=0)
    # Repeat each row, then flip to get variation along rows
    df_dy_analytic = np.tile(2 * deriv_2d_data.b * (deriv_2d_data.y - deriv_2d_data.y0),
                             (deriv_2d_data.f.shape[1], 1)).T
    assert_array_almost_equal(df_dy, df_dy_analytic, 5)


def test_first_derivative_too_small(deriv_1d_data):
    """Test first_derivative with too small an array."""
    with pytest.raises(ValueError):
        first_derivative(deriv_1d_data.values[None, :].T, x=deriv_1d_data.x, axis=1)


def test_first_derivative_scalar_delta():
    """Test first_derivative with a scalar passed for a delta."""
    df_dx = first_derivative(np.arange(3), delta=1)
    assert_array_almost_equal(df_dx, np.array([1., 1., 1.]), 6)


def test_first_derivative_masked():
    """Test that first_derivative properly propagates masks."""
    data = np.ma.arange(7)
    data[3] = np.ma.masked
    df_dx = first_derivative(data, delta=1)

    truth = np.ma.array([1., 1., 1., 1., 1., 1., 1.],
                        mask=[False, False, True, True, True, False, False])
    assert_array_almost_equal(df_dx, truth)
    assert_array_equal(df_dx.mask, truth.mask)


def test_first_derivative_masked_units():
    """Test that first_derivative properly propagates masks with units."""
    data = units('K') * np.ma.arange(7)
    data[3] = np.ma.masked
    x = units('m') * np.ma.arange(7)
    df_dx = first_derivative(data, x=x)

    truth = units('K / m') * np.ma.array(
        [1., 1., 1., 1., 1., 1., 1.],
        mask=[False, False, True, True, True, False, False])
    assert_array_almost_equal(df_dx, truth)
    assert_array_equal(df_dx.mask, truth.mask)


def test_second_derivative(deriv_1d_data):
    """Test second_derivative with a simple 1D array."""
    d2v_dx2 = second_derivative(deriv_1d_data.values, x=deriv_1d_data.x)

    # Worked by hand
    truth = np.ones_like(deriv_1d_data.values) * 0.2133333 * units('delta_degC/cm**2')
    assert_array_almost_equal(d2v_dx2, truth, 5)


def test_second_derivative_2d(deriv_2d_data):
    """Test second_derivative with a full 2D array."""
    df2_dx2 = second_derivative(deriv_2d_data.f, x=deriv_2d_data.x, axis=1)
    assert_array_almost_equal(df2_dx2,
                              np.ones_like(deriv_2d_data.f) * (2 * deriv_2d_data.a), 5)

    df2_dy2 = second_derivative(deriv_2d_data.f, x=deriv_2d_data.y, axis=0)
    assert_array_almost_equal(df2_dy2,
                              np.ones_like(deriv_2d_data.f) * (2 * deriv_2d_data.b), 5)


def test_second_derivative_too_small(deriv_1d_data):
    """Test second_derivative with too small an array."""
    with pytest.raises(ValueError):
        second_derivative(deriv_1d_data.values[None, :].T, x=deriv_1d_data.x, axis=1)


def test_second_derivative_scalar_delta():
    """Test second_derivative with a scalar passed for a delta."""
    df_dx = second_derivative(np.arange(3), delta=1)
    assert_array_almost_equal(df_dx, np.array([0., 0., 0.]), 6)


def test_laplacian(deriv_1d_data):
    """Test laplacian with simple 1D data."""
    laplac = laplacian(deriv_1d_data.values, coordinates=(deriv_1d_data.x,))

    # Worked by hand
    truth = np.ones_like(deriv_1d_data.values) * 0.2133333 * units('delta_degC/cm**2')
    assert_array_almost_equal(laplac, truth, 5)


def test_laplacian_2d(deriv_2d_data):
    """Test lapacian with full 2D arrays."""
    laplac_true = 2 * (np.ones_like(deriv_2d_data.f) * (deriv_2d_data.a + deriv_2d_data.b))
    laplac = laplacian(deriv_2d_data.f, coordinates=(deriv_2d_data.y, deriv_2d_data.x))
    assert_array_almost_equal(laplac, laplac_true, 5)


def test_parse_angle_abbrieviated():
    """Test abbrieviated directional text in degrees."""
    expected_angles_degrees = FULL_CIRCLE_DEGREES
    output_angles_degrees = parse_angle(DIR_STRS[:-1])
    assert_array_almost_equal(output_angles_degrees, expected_angles_degrees)


def test_parse_angle_ext():
    """Test extended (unabbrieviated) directional text in degrees."""
    test_dir_strs = ['NORTH', 'NORTHnorthEast', 'North_East', 'East__North_East',
                     'easT', 'east  south east', 'south east', ' south southeast',
                     'SOUTH', 'SOUTH SOUTH WEST', 'southWEST', 'WEST south_WEST',
                     'WeSt', 'WestNorth West', 'North West', 'NORTH north_WeSt']
    expected_angles_degrees = np.arange(0, 360, 22.5) * units.degree
    output_angles_degrees = parse_angle(test_dir_strs)
    assert_array_almost_equal(output_angles_degrees, expected_angles_degrees)


def test_parse_angle_mix_multiple():
    """Test list of extended (unabbrieviated) directional text in degrees in one go."""
    test_dir_strs = ['NORTH', 'nne', 'ne', 'east north east',
                     'easT', 'east  se', 'south east', ' south southeast',
                     'SOUTH', 'SOUTH SOUTH WEST', 'sw', 'WEST south_WEST',
                     'w', 'wnw', 'North West', 'nnw']
    expected_angles_degrees = FULL_CIRCLE_DEGREES
    output_angles_degrees = parse_angle(test_dir_strs)
    assert_array_almost_equal(output_angles_degrees, expected_angles_degrees)


def test_parse_angle_none():
    """Test list of extended (unabbrieviated) directional text in degrees in one go."""
    test_dir_strs = None
    expected_angles_degrees = np.nan
    output_angles_degrees = parse_angle(test_dir_strs)
    assert_array_almost_equal(output_angles_degrees, expected_angles_degrees)


def test_parse_angle_invalid_number():
    """Test list of extended (unabbrieviated) directional text in degrees in one go."""
    test_dir_strs = 365.
    expected_angles_degrees = np.nan
    output_angles_degrees = parse_angle(test_dir_strs)
    assert_array_almost_equal(output_angles_degrees, expected_angles_degrees)


def test_parse_angle_invalid_arr():
    """Test list of extended (unabbrieviated) directional text in degrees in one go."""
    test_dir_strs = ['nan', None, np.nan, 35, 35.5, 'north', 'andrewiscool']
    expected_angles_degrees = [np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan]
    output_angles_degrees = parse_angle(test_dir_strs)
    assert_array_almost_equal(output_angles_degrees, expected_angles_degrees)


def test_parse_angle_mix_multiple_arr():
    """Test list of extended (unabbrieviated) directional text in degrees in one go."""
    test_dir_strs = np.array(['NORTH', 'nne', 'ne', 'east north east',
                              'easT', 'east  se', 'south east', ' south southeast',
                              'SOUTH', 'SOUTH SOUTH WEST', 'sw', 'WEST south_WEST',
                              'w', 'wnw', 'North West', 'nnw'])
    expected_angles_degrees = FULL_CIRCLE_DEGREES
    output_angles_degrees = parse_angle(test_dir_strs)
    assert_array_almost_equal(output_angles_degrees, expected_angles_degrees)


def test_parse_angles_array():
    """Test array of angles to parse."""
    angles = np.array(['N', 'S', 'E', 'W'])
    expected_angles = np.array([0, 180, 90, 270]) * units.degree
    calculated_angles = parse_angle(angles)
    assert_array_almost_equal(calculated_angles, expected_angles)


def test_parse_angles_series():
    """Test pandas.Series of angles to parse."""
    angles = pd.Series(['N', 'S', 'E', 'W'])
    expected_angles = np.array([0, 180, 90, 270]) * units.degree
    calculated_angles = parse_angle(angles)
    assert_array_almost_equal(calculated_angles, expected_angles)


def test_parse_angles_single():
    """Test single input into `parse_angles`."""
    calculated_angle = parse_angle('SOUTH SOUTH EAST')
    expected_angle = 157.5 * units.degree
    assert_almost_equal(calculated_angle, expected_angle)


def test_gradient_2d(deriv_2d_data):
    """Test gradient with 2D arrays."""
    res = gradient(deriv_2d_data.f, coordinates=(deriv_2d_data.y, deriv_2d_data.x))
    truth = (np.array([[-0.25, -0.25, -0.25],
                       [1.75, 1.75, 1.75],
                       [4.75, 4.75, 4.75],
                       [5.75, 5.75, 5.75]]),
             np.array([[-3, -1, 4],
                       [-3, -1, 4],
                       [-3, -1, 4],
                       [-3, -1, 4]]))
    assert_array_almost_equal(res, truth, 5)


def test_gradient_4d(deriv_4d_data):
    """Test gradient with 4D arrays."""
    res = gradient(deriv_4d_data, deltas=(1, 1, 1, 1))
    truth = tuple(factor * np.ones_like(deriv_4d_data) for factor in (48., 16., 4., 1.))
    assert_array_almost_equal(res, truth, 8)


def test_gradient_restricted_axes(deriv_2d_data):
    """Test 2D gradient with 3D arrays and manual specification of axes."""
    res = gradient(deriv_2d_data.f[..., None], coordinates=(deriv_2d_data.y, deriv_2d_data.x),
                   axes=(0, 1))
    truth = (np.array([[[-0.25], [-0.25], [-0.25]],
                       [[1.75], [1.75], [1.75]],
                       [[4.75], [4.75], [4.75]],
                       [[5.75], [5.75], [5.75]]]),
             np.array([[[-3], [-1], [4]],
                       [[-3], [-1], [4]],
                       [[-3], [-1], [4]],
                       [[-3], [-1], [4]]]))
    assert_array_almost_equal(res, truth, 5)


def test_bounding_indices():
    """Test finding bounding indices."""
    data = np.array([[1, 2, 3, 1], [5, 6, 7, 8]])
    above, below, good = find_bounding_indices(data, [1.5, 7], axis=1, from_below=True)

    assert_array_equal(above[1], np.array([[1, 0], [0, 3]]))
    assert_array_equal(below[1], np.array([[0, -1], [-1, 2]]))
    assert_array_equal(good, np.array([[True, False], [False, True]]))


def test_bounding_indices_above():
    """Test finding bounding indices from above."""
    data = np.array([[1, 2, 3, 1], [5, 6, 7, 8]])
    above, below, good = find_bounding_indices(data, [1.5, 7], axis=1, from_below=False)

    assert_array_equal(above[1], np.array([[3, 0], [0, 3]]))
    assert_array_equal(below[1], np.array([[2, -1], [-1, 2]]))
    assert_array_equal(good, np.array([[True, False], [False, True]]))


def test_angle_to_direction():
    """Test single angle in degree."""
    expected_dirs = DIR_STRS[:-1]  # UND at -1
    output_dirs = [angle_to_direction(angle) for angle in FULL_CIRCLE_DEGREES]
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_edge():
    """Test single angle edge case (360 and no units) in degree."""
    expected_dirs = 'N'
    output_dirs = angle_to_direction(360)
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_list():
    """Test list of angles in degree."""
    expected_dirs = DIR_STRS[:-1]
    output_dirs = list(angle_to_direction(FULL_CIRCLE_DEGREES))
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_arr():
    """Test array of angles in degree."""
    expected_dirs = DIR_STRS[:-1]
    output_dirs = angle_to_direction(FULL_CIRCLE_DEGREES)
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_full():
    """Test the `full` keyword argument, expecting unabbrieviated output."""
    expected_dirs = [
        'North', 'North North East', 'North East', 'East North East',
        'East', 'East South East', 'South East', 'South South East',
        'South', 'South South West', 'South West', 'West South West',
        'West', 'West North West', 'North West', 'North North West'
    ]
    output_dirs = angle_to_direction(FULL_CIRCLE_DEGREES, full=True)
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_invalid_scalar():
    """Test invalid angle."""
    expected_dirs = UND
    output_dirs = angle_to_direction(None)
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_invalid_arr():
    """Test array of invalid angles."""
    expected_dirs = ['NE', UND, UND, UND, 'N']
    output_dirs = angle_to_direction(['46', None, np.nan, None, '362.'])
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_level_4():
    """Test non-existent level of complexity."""
    with pytest.raises(ValueError) as exc:
        angle_to_direction(FULL_CIRCLE_DEGREES, level=4)
    assert 'cannot be less than 1 or greater than 3' in str(exc.value)


def test_angle_to_direction_level_3():
    """Test array of angles in degree."""
    expected_dirs = DIR_STRS[:-1]  # UND at -1
    output_dirs = angle_to_direction(FULL_CIRCLE_DEGREES, level=3)
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_level_2():
    """Test array of angles in degree."""
    expected_dirs = [
        'N', 'N', 'NE', 'NE', 'E', 'E', 'SE', 'SE',
        'S', 'S', 'SW', 'SW', 'W', 'W', 'NW', 'NW'
    ]
    output_dirs = angle_to_direction(FULL_CIRCLE_DEGREES, level=2)
    assert_array_equal(output_dirs, expected_dirs)


def test_angle_to_direction_level_1():
    """Test array of angles in degree."""
    expected_dirs = [
        'N', 'N', 'N', 'E', 'E', 'E', 'E', 'S', 'S', 'S', 'S',
        'W', 'W', 'W', 'W', 'N']
    output_dirs = angle_to_direction(FULL_CIRCLE_DEGREES, level=1)
    assert_array_equal(output_dirs, expected_dirs)


def test_azimuth_range_to_lat_lon():
    """Test conversion of azimuth and range to lat/lon grid."""
    az = [332.2403, 334.6765, 337.2528, 339.73846, 342.26257]
    rng = [2125., 64625., 127125., 189625., 252125., 314625.]
    clon = -89.98416666666667
    clat = 32.27972222222222
    with pytest.warns(UserWarning, match='not a Pint-Quantity'):
        output_lon, output_lat = azimuth_range_to_lat_lon(az, rng, clon, clat)
    true_lon = [[-89.9946968, -90.3061798, -90.6211612, -90.9397425, -91.2620282,
                -91.5881257],
                [-89.9938369, -90.2799198, -90.5692874, -90.8620385, -91.1582743,
                -91.4580996],
                [-89.9929086, -90.251559, -90.5132417, -90.7780507, -91.0460827,
                -91.3174374],
                [-89.9919961, -90.2236737, -90.4581161, -90.6954113, -90.9356497,
                -91.178925],
                [-89.9910545, -90.1948876, -90.4011921, -90.6100481, -90.8215385,
                -91.0357492]]
    true_lat = [[32.2966329, 32.7936114, 33.2898102, 33.7852055, 34.2797726,
                34.773486],
                [32.2969961, 32.804717, 33.3117799, 33.8181643, 34.3238488,
                34.8288114],
                [32.2973461, 32.8154229, 33.3329617, 33.8499452, 34.3663556,
                34.8821746],
                [32.29765, 32.8247204, 33.3513589, 33.8775516, 34.4032838,
                34.9285404],
                [32.2979242, 32.8331062, 33.367954, 33.9024562, 34.4366016,
                34.9703782]]
    assert_array_almost_equal(output_lon, true_lon, 6)
    assert_array_almost_equal(output_lat, true_lat, 6)


def test_azimuth_range_to_lat_lon_diff_ellps():
    """Test conversion of azimuth and range to lat/lon grid."""
    az = [332.2403, 334.6765, 337.2528, 339.73846, 342.26257] * units.degrees
    rng = [2125., 64625., 127125., 189625., 252125., 314625.] * units.meters
    clon = -89.98416666666667
    clat = 32.27972222222222
    output_lon, output_lat = azimuth_range_to_lat_lon(az, rng, clon, clat, Geod(ellps='WGS84'))
    true_lon = [[-89.9946749, -90.3055083, -90.6198256, -90.9377279, -91.2593193,
                -91.5847066],
                [-89.9938168, -90.279303, -90.5680603, -90.860187, -91.1557841,
                -91.4549558],
                [-89.9928904, -90.2510012, -90.5121319, -90.7763758, -91.0438294,
                -91.3145919],
                [-89.9919799, -90.2231741, -90.4571217, -90.6939102, -90.9336298,
                -91.1763737],
                [-89.9910402, -90.194448, -90.4003169, -90.6087268, -90.8197603,
                -91.0335027]]
    true_lat = [[32.2966791, 32.794996, 33.2924932, 33.7891466, 34.2849315,
                34.7798223],
                [32.2970433, 32.8061309, 33.3145188, 33.8221862, 34.3291116,
                34.835273],
                [32.2973942, 32.816865, 33.3357544, 33.8540448, 34.3717184,
                34.8887564],
                [32.297699, 32.826187, 33.3541984, 33.8817186, 34.4087331,
                34.9352264],
                [32.2979739, 32.834595, 33.3708355, 33.906684, 34.4421288,
                34.9771578]]
    assert_array_almost_equal(output_lon, true_lon, 6)
    assert_array_almost_equal(output_lat, true_lat, 6)


def test_3d_gradient_3d_data_no_axes(deriv_4d_data):
    """Test 3D gradient with 3D data and no axes parameter."""
    test = deriv_4d_data[0]
    res = gradient(test, deltas=(1, 1, 1))
    truth = tuple(factor * np.ones_like(test) for factor in (16., 4., 1.))
    assert_array_almost_equal(res, truth, 8)


def test_2d_gradient_3d_data_no_axes(deriv_4d_data):
    """Test for failure of 2D gradient with 3D data and no axes parameter."""
    test = deriv_4d_data[0]
    with pytest.raises(ValueError) as exc:
        gradient(test, deltas=(1, 1))
    assert 'must match the number of dimensions' in str(exc.value)


def test_3d_gradient_2d_data_no_axes(deriv_4d_data):
    """Test for failure of 3D gradient with 2D data and no axes parameter."""
    test = deriv_4d_data[0, 0]
    with pytest.raises(ValueError) as exc:
        gradient(test, deltas=(1, 1, 1))
    assert 'must match the number of dimensions' in str(exc.value)


def test_2d_gradient_4d_data_2_axes_3_deltas(deriv_4d_data):
    """Test 2D gradient of 4D data with 2 axes and 3 deltas."""
    res = gradient(deriv_4d_data, deltas=(1, 1, 1), axes=(-2, -1))
    truth = tuple(factor * np.ones_like(deriv_4d_data) for factor in (4., 1.))
    assert_array_almost_equal(res, truth, 8)


def test_2d_gradient_4d_data_2_axes_2_deltas(deriv_4d_data):
    """Test 2D gradient of 4D data with 2 axes and 2 deltas."""
    res = gradient(deriv_4d_data, deltas=(1, 1), axes=(0, 1))
    truth = tuple(factor * np.ones_like(deriv_4d_data) for factor in (48., 16.))
    assert_array_almost_equal(res, truth, 8)


def test_2d_gradient_4d_data_2_axes_1_deltas(deriv_4d_data):
    """Test for failure of 2D gradient of 4D data with 2 axes and 1 deltas."""
    with pytest.raises(ValueError) as exc:
        gradient(deriv_4d_data, deltas=(1, ), axes=(1, 2))
    assert 'cannot be less than that of "axes"' in str(exc.value)


@pytest.mark.parametrize('geog_data', ('+proj=lcc lat_1=25', '+proj=latlon', '+proj=stere'),
                         indirect=True)
def test_geospatial_gradient_geographic(geog_data):
    """Test geospatial_gradient on geographic coordinates."""
    # Generate a field of temperature on a lat/lon grid
    crs, lons, lats, _, arr, mx, my, dx, dy = geog_data
    grad_x, grad_y = geospatial_gradient(arr, longitude=lons, latitude=lats, crs=crs)

    # Calculate the true fields using known map-correct approach
    truth_x = mx * first_derivative(arr, delta=dx, axis=1)
    truth_y = my * first_derivative(arr, delta=dy, axis=0)

    assert_array_almost_equal(grad_x, truth_x)
    assert_array_almost_equal(grad_y, truth_y)


@pytest.mark.parametrize('return_only,length', [(None, 2), ('df/dx', 3), (('df/dx',), 1)])
def test_geospatial_gradient_return_subset(return_only, length):
    """Test geospatial_gradient's return_only as string and tuple subset."""
    a = np.arange(4)[None, :]
    arr = np.r_[a, a, a] * units('m/s')
    lons = np.array([-100, -90, -80, -70]) * units('degree')
    lats = np.array([45, 55, 65]) * units('degree')
    crs = CRS('+proj=latlon')

    ddx = geospatial_gradient(
        arr, longitude=lons, latitude=lats, crs=crs, return_only=return_only)

    assert len(ddx) == length


def test_first_derivative_xarray_lonlat(test_da_lonlat):
    """Test first derivative with an xarray.DataArray on a lonlat grid in each axis usage."""
    deriv = first_derivative(test_da_lonlat, axis='lon')  # dimension coordinate name
    deriv_alt1 = first_derivative(test_da_lonlat, axis='x')  # axis type
    deriv_alt2 = first_derivative(test_da_lonlat, axis=-1)  # axis number

    # Build the xarray of the desired values
    partial = xr.DataArray(
        np.array([-3.30782978e-06, -3.42816074e-06, -3.57012948e-06, -3.73759364e-06]),
        coords={'lat': test_da_lonlat['lat']},
        dims=('lat',)
    )
    _, truth = xr.broadcast(test_da_lonlat, partial)
    truth.coords['metpy_crs'] = test_da_lonlat['metpy_crs']
    truth.attrs['units'] = 'kelvin / meter'
    truth = truth.metpy.quantify()

    # Assert result matches expectation
    xr.testing.assert_allclose(deriv, truth)
    assert deriv.metpy.units == truth.metpy.units

    # Assert alternative specifications give same result
    xr.testing.assert_identical(deriv_alt1, deriv)
    xr.testing.assert_identical(deriv_alt2, deriv)


def test_first_derivative_xarray_time_and_default_axis(test_da_xy):
    """Test first derivative with an xarray.DataArray over time as default first dimension."""
    deriv = first_derivative(test_da_xy)
    truth = xr.full_like(test_da_xy, -0.000777000777)
    truth.attrs['units'] = 'kelvin / second'
    truth = truth.metpy.quantify()

    xr.testing.assert_allclose(deriv, truth)
    assert deriv.metpy.units == truth.metpy.units


def test_first_derivative_xarray_time_subsecond_precision():
    """Test time derivative with an xarray.DataArray where subsecond precision is needed."""
    test_da = xr.DataArray([299.5, 300, 300.5],
                           dims='time',
                           coords={'time': np.array(['2019-01-01T00:00:00.0',
                                                     '2019-01-01T00:00:00.1',
                                                     '2019-01-01T00:00:00.2'],
                                                    dtype='datetime64[ms]')},
                           attrs={'units': 'kelvin'})

    deriv = first_derivative(test_da)

    truth = xr.full_like(test_da, 5.)
    truth.attrs['units'] = 'kelvin / second'
    truth = truth.metpy.quantify()

    xr.testing.assert_allclose(deriv, truth)
    assert deriv.metpy.units == truth.metpy.units


def test_second_derivative_xarray_lonlat(test_da_lonlat):
    """Test second derivative with an xarray.DataArray on a lonlat grid."""
    deriv = second_derivative(test_da_lonlat, axis='lat')

    # Build the xarray of the desired values
    partial = xr.DataArray(
        np.array([1.67155420e-14, 1.67155420e-14, 1.74268211e-14, 1.74268211e-14]),
        coords={'lat': test_da_lonlat['lat']},
        dims=('lat',)
    )
    _, truth = xr.broadcast(test_da_lonlat, partial)
    truth.coords['metpy_crs'] = test_da_lonlat['metpy_crs']
    truth.attrs['units'] = 'kelvin / meter^2'
    truth = truth.metpy.quantify()

    xr.testing.assert_allclose(deriv, truth)
    assert deriv.metpy.units == truth.metpy.units


def test_gradient_xarray(test_da_xy):
    """Test the 3D gradient calculation with a 4D DataArray in each axis usage."""
    deriv_x, deriv_y, deriv_p = gradient(test_da_xy, axes=('x', 'y', 'isobaric'))
    deriv_x_alt1, deriv_y_alt1, deriv_p_alt1 = gradient(test_da_xy,
                                                        axes=('x', 'y', 'vertical'))
    deriv_x_alt2, deriv_y_alt2, deriv_p_alt2 = gradient(test_da_xy, axes=(3, 2, 1))

    truth_x = xr.full_like(test_da_xy, -6.993007e-07)
    truth_x.attrs['units'] = 'kelvin / meter'
    truth_x = truth_x.metpy.quantify()

    truth_y = xr.full_like(test_da_xy, -2.797203e-06)
    truth_y.attrs['units'] = 'kelvin / meter'
    truth_y = truth_y.metpy.quantify()

    partial = xr.DataArray(
        np.array([0.04129204, 0.03330003, 0.02264402]),
        coords={'isobaric': test_da_xy['isobaric']},
        dims=('isobaric',)
    )
    _, truth_p = xr.broadcast(test_da_xy, partial)
    truth_p.coords['metpy_crs'] = test_da_xy['metpy_crs']
    truth_p.attrs['units'] = 'kelvin / hectopascal'
    truth_p = truth_p.metpy.quantify()

    # Assert results match expectations
    xr.testing.assert_allclose(deriv_x, truth_x)
    assert deriv_x.metpy.units == truth_x.metpy.units
    xr.testing.assert_allclose(deriv_y, truth_y)
    assert deriv_y.metpy.units == truth_y.metpy.units
    xr.testing.assert_allclose(deriv_p, truth_p)
    assert deriv_p.metpy.units == truth_p.metpy.units

    # Assert alternative specifications give same results (up to attribute differences)
    xr.testing.assert_equal(deriv_x_alt1, deriv_x)
    xr.testing.assert_equal(deriv_y_alt1, deriv_y)
    xr.testing.assert_equal(deriv_p_alt1, deriv_p)
    xr.testing.assert_equal(deriv_x_alt2, deriv_x)
    xr.testing.assert_equal(deriv_y_alt2, deriv_y)
    xr.testing.assert_equal(deriv_p_alt2, deriv_p)


def test_gradient_xarray_implicit_axes(test_da_xy):
    """Test the 2D gradient calculation with a 2D DataArray and no axes specified."""
    data = test_da_xy.isel(time=0, isobaric=2)
    deriv_y, deriv_x = gradient(data)

    truth_x = xr.full_like(data, -6.993007e-07)
    truth_x.attrs['units'] = 'kelvin / meter'
    truth_x = truth_x.metpy.quantify()

    truth_y = xr.full_like(data, -2.797203e-06)
    truth_y.attrs['units'] = 'kelvin / meter'
    truth_y = truth_y.metpy.quantify()

    xr.testing.assert_allclose(deriv_x, truth_x)
    assert deriv_x.metpy.units == truth_x.metpy.units

    xr.testing.assert_allclose(deriv_y, truth_y)
    assert deriv_y.metpy.units == truth_y.metpy.units


def test_gradient_xarray_implicit_axes_transposed(test_da_lonlat):
    """Test the 2D gradient with no axes specified but in x/y order."""
    test_da = test_da_lonlat.isel(isobaric=0).transpose('lon', 'lat')
    deriv_x, deriv_y = gradient(test_da)

    truth_x = xr.DataArray(
        np.array(
            [[-3.30782978e-06, -3.42816074e-06, -3.57012948e-06, -3.73759364e-06],
             [-3.30782978e-06, -3.42816074e-06, -3.57012948e-06, -3.73759364e-06],
             [-3.30782978e-06, -3.42816074e-06, -3.57012948e-06, -3.73759364e-06],
             [-3.30782978e-06, -3.42816074e-06, -3.57012948e-06, -3.73759364e-06]]
        ) * units('kelvin / meter'),
        dims=test_da.dims,
        coords=test_da.coords
    )
    truth_y = xr.DataArray(
        np.array(
            [[-1.15162805e-05, -1.15101023e-05, -1.15037894e-05, -1.14973413e-05],
             [-1.15162805e-05, -1.15101023e-05, -1.15037894e-05, -1.14973413e-05],
             [-1.15162805e-05, -1.15101023e-05, -1.15037894e-05, -1.14973413e-05],
             [-1.15162805e-05, -1.15101023e-05, -1.15037894e-05, -1.14973413e-05]]
        ) * units('kelvin / meter'),
        dims=test_da.dims,
        coords=test_da.coords
    )

    xr.testing.assert_allclose(deriv_x, truth_x)
    assert deriv_x.metpy.units == truth_x.metpy.units

    xr.testing.assert_allclose(deriv_y, truth_y)
    assert deriv_y.metpy.units == truth_y.metpy.units


def test_laplacian_xarray_lonlat(test_da_lonlat):
    """Test laplacian with an xarray.DataArray on a lonlat grid."""
    laplac = laplacian(test_da_lonlat, axes=('lat', 'lon'))

    # Build the xarray of the desired values
    partial = xr.DataArray(
        np.array([1.67155420e-14, 1.67155420e-14, 1.74268211e-14, 1.74268211e-14]),
        coords={'lat': test_da_lonlat['lat']},
        dims=('lat',)
    )
    _, truth = xr.broadcast(test_da_lonlat, partial)
    truth.coords['metpy_crs'] = test_da_lonlat['metpy_crs']
    truth.attrs['units'] = 'kelvin / meter^2'
    truth = truth.metpy.quantify()

    xr.testing.assert_allclose(laplac, truth)
    assert laplac.metpy.units == truth.metpy.units


def test_first_derivative_xarray_pint_conversion(test_da_lonlat):
    """Test first derivative with implicit xarray to pint quantity conversion."""
    dx, _ = grid_deltas_from_dataarray(test_da_lonlat)
    deriv = first_derivative(test_da_lonlat, delta=dx, axis=-1)
    truth = np.array([[[-3.30782978e-06] * 4, [-3.42816074e-06] * 4, [-3.57012948e-06] * 4,
                       [-3.73759364e-06] * 4]] * 3) * units('kelvin / meter')
    assert_array_almost_equal(deriv, truth, 12)


def test_gradient_xarray_pint_conversion(test_da_xy):
    """Test the 2D gradient calculation with a 2D DataArray and implicit pint conversion."""
    data = test_da_xy.isel(time=0, isobaric=2)
    deriv_y, deriv_x = gradient(data, coordinates=(data.metpy.y, data.metpy.x))

    truth_x = np.ones_like(data) * -6.993007e-07 * units('kelvin / meter')
    truth_y = np.ones_like(data) * -2.797203e-06 * units('kelvin / meter')

    assert_array_almost_equal(deriv_x, truth_x, 12)
    assert_array_almost_equal(deriv_y, truth_y, 12)


def test_remove_nans():
    """Test removal of NaNs."""
    x = np.array([3, 2, np.nan, 5, 6, np.nan])
    y = np.arange(0, len(x))
    y_test, x_test = _remove_nans(y, x)
    x_expected = np.array([3, 2, 5, 6])
    y_expected = np.array([0, 1, 3, 4])
    assert_array_almost_equal(x_expected, x_test, 0)
    assert_almost_equal(y_expected, y_test, 0)


@pytest.mark.parametrize('subset', (False, True))
@pytest.mark.parametrize('datafile, assign_lat_lon, no_crs, transpose',
                         [('GFS_test.nc', False, False, False),
                          ('GFS_test.nc', False, True, False),
                          ('NAM_test.nc', False, False, False),
                          ('NAM_test.nc', True, False, False),
                          ('NAM_test.nc', True, False, True)])
def test_parse_grid_arguments_xarray(datafile, assign_lat_lon, no_crs, transpose, subset):
    """Test the operation of parse_grid_arguments with xarray data."""
    @parse_grid_arguments
    @preprocess_and_wrap(broadcast=['scalar', 'parallel_scale', 'meridional_scale'],
                         wrap_like=('scalar', 'dx', 'dy', 'scalar', 'scalar', 'latitude',
                                    None, None))
    def check_params(scalar, dx=None, dy=None, parallel_scale=None, meridional_scale=None,
                     latitude=None, x_dim=-1, y_dim=-2):
        return scalar, dx, dy, parallel_scale, meridional_scale, latitude, x_dim, y_dim

    data = xr.open_dataset(get_test_data(datafile, as_file_obj=False))

    if no_crs:
        data = data.drop_vars(('LatLon_Projection',))
        temp = data.Temperature_isobaric
    else:
        temp = data.metpy.parse_cf('Temperature_isobaric')

    if transpose:
        temp = temp.transpose(..., 'x', 'y')

    if assign_lat_lon:
        temp = temp.metpy.assign_latitude_longitude()
    if subset:
        temp = temp.isel(time=0).metpy.sel(vertical=500 * units.hPa)

    t, dx, dy, p, m, lat, x_dim, y_dim = check_params(temp)

    if transpose:
        if subset:
            assert x_dim == 0
            assert y_dim == 1
        else:
            assert x_dim == 2
            assert y_dim == 3
    elif subset:
        assert x_dim == 1
        assert y_dim == 0
    else:
        assert x_dim == 3
        assert y_dim == 2

    assert_array_equal(t, temp)

    assert p.shape == t.shape
    assert_array_equal(p.metpy.x, t.metpy.x)
    assert_array_equal(p.metpy.y, t.metpy.y)

    assert m.shape == t.shape
    assert_array_equal(m.metpy.x, t.metpy.x)
    assert_array_equal(m.metpy.y, t.metpy.y)

    assert dx.check('m')
    assert dy.check('m')

    assert_array_almost_equal(lat, data.lat, 5)


@pytest.mark.parametrize('xy_order', (False, True))
def test_parse_grid_arguments_cartesian(test_da_xy, xy_order):
    """Test the operation of parse_grid_arguments with no lat/lon info."""
    @parse_grid_arguments
    @preprocess_and_wrap(broadcast=['scalar', 'parallel_scale', 'meridional_scale'],
                         wrap_like=('scalar', 'dx', 'dy', 'scalar', 'scalar', 'latitude',
                                    None, None))
    def check_params(scalar, dx=None, dy=None, x_dim=-1, y_dim=-2,
                     parallel_scale=None, meridional_scale=None, latitude=None):
        return scalar, dx, dy, parallel_scale, meridional_scale, latitude, x_dim, y_dim

    # Remove CRS from dataarray
    data = test_da_xy.reset_coords('metpy_crs', drop=True)
    del data.attrs['grid_mapping']

    if xy_order:
        data = data.transpose(..., 'x', 'y')

    t, dx, dy, p, m, lat, x_dim, y_dim = check_params(data)
    if xy_order:
        assert x_dim == 2
        assert y_dim == 3
    else:
        assert x_dim == 3
        assert y_dim == 2

    assert_array_almost_equal(t, data)
    assert_array_almost_equal(dx, 500 * units.km)
    assert_array_almost_equal(dy, 500 * units.km)

    assert p is None
    assert m is None
    assert lat is None


def test_parse_grid_arguments_missing_coords():
    """Test parse_grid_arguments with data with missing dimension coordinates."""
    @parse_grid_arguments
    @preprocess_and_wrap()
    def check_params(scalar, dx=None, dy=None, x_dim=-1, y_dim=-2):
        """Test parameter passing and filling."""

    lat, lon = np.meshgrid(np.array([38., 40., 42]), np.array([263., 265., 267.]))
    test_da = xr.DataArray(
        np.linspace(300, 250, 3 * 3).reshape((3, 3)),
        name='temperature',
        dims=('dim_0', 'dim_1'),
        coords={
            'lat': xr.DataArray(lat, dims=('dim_0', 'dim_1'),
                                attrs={'units': 'degrees_north'}),
            'lon': xr.DataArray(lon, dims=('dim_0', 'dim_1'), attrs={'units': 'degrees_east'})
        },
        attrs={'units': 'K'}).to_dataset().metpy.parse_cf('temperature')

    with pytest.raises(AttributeError,
                       match='horizontal dimension coordinates cannot be found.'):
        check_params(test_da)


def test_parse_grid_arguments_unknown_dims():
    """Test parse_grid_arguments with data with unknown dimensions."""
    @parse_grid_arguments
    @preprocess_and_wrap(broadcast=['scalar', 'parallel_scale', 'meridional_scale'])
    def check_params(scalar, dx=None, dy=None, x_dim=-1, y_dim=-2, parallel_scale=None,
                     meridional_scale=None, latitude=None):
        return x_dim, y_dim

    dim0 = np.arange(3)
    dim1 = np.arange(5, 11, 2)
    test_da = xr.DataArray(
        np.linspace(300, 250, 3 * 3).reshape((3, 3)),
        name='temperature',
        dims=('dim_0', 'dim_1'),
        coords={
            'dim_0': xr.DataArray(dim0, dims=('dim_0',), attrs={'units': 'm'}),
            'dim_1': xr.DataArray(dim1, dims=('dim_1',), attrs={'units': 'm'}),
        },
        attrs={'units': 'K'}).to_dataset().metpy.parse_cf('temperature')

    with pytest.warns(UserWarning,
                      match='Horizontal dimension numbers not found.'):
        x_dim, y_dim = check_params(test_da, dx=2.0 * units.m, dy=1.0 * units.m)
        assert y_dim == -2
        assert x_dim == -1


# Ported from original test for add_grid_arguments_from_xarray
def test_parse_grid_arguments_from_dataarray():
    """Test the parse grid arguments decorator for adding in arguments from xarray."""
    @parse_grid_arguments
    def return_the_kwargs(
        da,
        dz=None,
        dy=None,
        dx=None,
        vertical_dim=None,
        y_dim=None,
        x_dim=None,
        latitude=None,
        parallel_scale=None,
        meridional_scale=None
    ):
        return {
            'dz': dz,
            'dy': dy,
            'dx': dx,
            'vertical_dim': vertical_dim,
            'y_dim': y_dim,
            'x_dim': x_dim,
            'latitude': latitude
        }

    data = xr.DataArray(
        np.zeros((1, 2, 2, 2)),
        dims=('time', 'isobaric', 'lat', 'lon'),
        coords={
            'time': ['2020-01-01T00:00Z'],
            'isobaric': (('isobaric',), [850., 700.], {'units': 'hPa'}),
            'lat': (('lat',), [30., 40.], {'units': 'degrees_north'}),
            'lon': (('lon',), [-100., -90.], {'units': 'degrees_east'})
        }
    ).to_dataset(name='zeros').metpy.parse_cf('zeros')
    result = return_the_kwargs(data)
    assert_array_almost_equal(result['dz'], [-150.] * units.hPa)
    assert_array_almost_equal(result['dy'], 1109415.632 * units.meter, 2)
    assert_array_almost_equal(result['dx'], 1113194.90793274 * units.meter, 2)
    assert result['vertical_dim'] == 1
    assert result['y_dim'] == 2
    assert result['x_dim'] == 3
    assert_array_almost_equal(
        result['latitude'].metpy.unit_array,
        [30., 40.] * units.degrees_north
    )
    # Verify latitude is xarray so can be broadcast,
    # see https://github.com/Unidata/MetPy/pull/1490#discussion_r483198245
    assert isinstance(result['latitude'], xr.DataArray)


def test_nominal_grid_deltas():
    """Test nominal_lat_lon_grid_deltas with basic params and non-default Geod."""
    lat = np.array([25., 35., 45.]) * units.degree
    lon = np.array([-105, -100, -95, -90]) * units.degree

    dx, dy = nominal_lat_lon_grid_deltas(lon, lat, Geod(a=4370997))
    assert_array_almost_equal(dx, 381441.44622397297 * units.m)
    assert_array_almost_equal(dy, [762882.89244795, 762882.89244795] * units.m)


def test_nominal_grid_deltas_trivial_nd():
    """Test that we can pass arrays with only one real dimension."""
    lat = np.array([25., 35., 45.]).reshape(1, 1, -1, 1) * units.degree
    lon = np.array([-105, -100, -95, -90]).reshape(1, 1, 1, -1) * units.degree

    dx, dy = nominal_lat_lon_grid_deltas(lon, lat)
    assert_array_almost_equal(dx, 556597.45396637 * units.m)
    assert_array_almost_equal(dy, [1108538.7325489, 1110351.4762828] * units.m)


def test_nominal_grid_deltas_raises():
    """Test that nominal_lat_lon_grid_deltas raises with full 2D inputs."""
    lat = np.array([[25.] * 4, [35.] * 4, [45.] * 4])
    lon = np.array([[-105, -100, -95, -90]] * 3)
    with pytest.raises(ValueError, match='one dimensional'):
        nominal_lat_lon_grid_deltas(lon, lat)


@pytest.mark.parametrize('return_only,length', [(None, 4),
                                                ('du/dx', 3),
                                                (('du/dx', 'dv/dy'), 2),
                                                (('du/dx',), 1)])
def test_vector_derivative_return_subset(return_only, length):
    """Test vector_derivative's return_only as string and tuple subset."""
    a = np.arange(4)[None, :]
    u = v = np.r_[a, a, a] * units('m/s')
    lons = np.array([-100, -90, -80, -70]) * units('degree')
    lats = np.array([45, 55, 65]) * units('degree')
    crs = CRS('+proj=latlon')

    ddx = vector_derivative(
        u, v, longitude=lons, latitude=lats, crs=crs, return_only=return_only)

    assert len(ddx) == length
