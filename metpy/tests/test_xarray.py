# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the operation of MetPy's XArray accessors."""
from __future__ import absolute_import

from collections import OrderedDict

import cartopy.crs as ccrs
import numpy as np
import pytest
import xarray as xr

from metpy.testing import assert_almost_equal, assert_array_equal, get_test_data
from metpy.units import units
from metpy.xarray import check_matching_coordinates, preprocess_xarray


# Seed RandomState for deterministic tests
np.random.seed(81964262)


@pytest.fixture
def test_ds():
    """Provide an xarray dataset for testing."""
    return xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))


@pytest.fixture
def test_ds_generic():
    """Provide a generic-coordinate dataset for testing."""
    return xr.DataArray(np.random.random((1, 3, 3, 5, 5)),
                        coords={
                            'a': xr.DataArray(np.arange(1), dims='a'),
                            'b': xr.DataArray(np.arange(3), dims='b'),
                            'c': xr.DataArray(np.arange(3), dims='c'),
                            'd': xr.DataArray(np.arange(5), dims='d'),
                            'e': xr.DataArray(np.arange(5), dims='e')
    }, dims=['a', 'b', 'c', 'd', 'e'], name='test').to_dataset()


@pytest.fixture
def test_var(test_ds):
    """Provide a standard, parsed, variable for testing."""
    return test_ds.metpy.parse_cf('Temperature')


def test_projection(test_var):
    """Test getting the proper projection out of the variable."""
    crs = test_var.metpy.crs
    assert crs['grid_mapping_name'] == 'lambert_conformal_conic'

    assert isinstance(test_var.metpy.cartopy_crs, ccrs.LambertConformal)


def test_no_projection(test_ds):
    """Test getting the crs attribute when not available produces a sensible error."""
    var = test_ds.lat
    with pytest.raises(AttributeError) as exc:
        var.metpy.crs

    assert 'not available' in str(exc.value)


def test_globe(test_var):
    """Test getting the globe belonging to the projection."""
    globe = test_var.metpy.cartopy_globe

    assert globe.to_proj4_params() == OrderedDict([('ellps', 'sphere'),
                                                   ('a', 6367470.21484375),
                                                   ('b', 6367470.21484375)])
    assert isinstance(globe, ccrs.Globe)


def test_unit_array(test_var):
    """Test unit handling through the accessor."""
    arr = test_var.metpy.unit_array
    assert isinstance(arr, units.Quantity)
    assert arr.units == units.kelvin


def test_units(test_var):
    """Test the units property on the accessor."""
    assert test_var.metpy.units == units('kelvin')


def test_convert_units(test_var):
    """Test in-place conversion of units."""
    test_var.metpy.convert_units('degC')

    # Check that variable metadata is updated
    assert test_var.attrs['units'] == 'degC'

    # Make sure we now get an array back with properly converted values
    assert test_var.metpy.unit_array.units == units.degC
    assert_almost_equal(test_var[0, 0, 0, 0], 18.44 * units.degC, 2)


def test_radian_projection_coords():
    """Test fallback code for radian units in projection coordinate variables."""
    proj = xr.DataArray(0, attrs={'grid_mapping_name': 'geostationary',
                                  'perspective_point_height': 3})
    x = xr.DataArray(np.arange(3),
                     attrs={'standard_name': 'projection_x_coordinate', 'units': 'radians'})
    y = xr.DataArray(np.arange(2),
                     attrs={'standard_name': 'projection_y_coordinate', 'units': 'radians'})
    data = xr.DataArray(np.arange(6).reshape(2, 3), coords=(y, x), dims=('y', 'x'),
                        attrs={'grid_mapping': 'fixedgrid_projection'})
    ds = xr.Dataset({'data': data, 'fixedgrid_projection': proj})

    # Check that the coordinates in this case are properly converted
    data_var = ds.metpy.parse_cf('data')
    assert data_var.coords['x'].metpy.unit_array[1] == 3 * units.meter
    assert data_var.coords['y'].metpy.unit_array[1] == 3 * units.meter


def test_missing_grid_mapping():
    """Test falling back to implicit lat/lon projection."""
    lon = xr.DataArray(-np.arange(3),
                       attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat = xr.DataArray(np.arange(2),
                       attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    data = xr.DataArray(np.arange(6).reshape(2, 3), coords=(lat, lon), dims=('y', 'x'))
    ds = xr.Dataset({'data': data})

    data_var = ds.metpy.parse_cf('data')
    assert 'crs' in data_var.coords


def test_missing_grid_mapping_var(caplog):
    """Test behavior when we can't find the variable pointed to by grid_mapping."""
    x = xr.DataArray(np.arange(3),
                     attrs={'standard_name': 'projection_x_coordinate', 'units': 'radians'})
    y = xr.DataArray(np.arange(2),
                     attrs={'standard_name': 'projection_y_coordinate', 'units': 'radians'})
    data = xr.DataArray(np.arange(6).reshape(2, 3), coords=(y, x), dims=('y', 'x'),
                        attrs={'grid_mapping': 'fixedgrid_projection'})
    ds = xr.Dataset({'data': data})

    ds.metpy.parse_cf('data')  # Should log a warning

    for record in caplog.records:
        assert record.levelname == 'WARNING'
    assert 'Could not find' in caplog.text


def test_preprocess_xarray():
    """Test xarray preprocessing decorator."""
    data = xr.DataArray(np.ones(3), attrs={'units': 'km'})
    data2 = xr.DataArray(np.ones(3), attrs={'units': 'm'})

    @preprocess_xarray
    def func(a, b):
        return a.to('m') + b

    assert_array_equal(func(data, b=data2), np.array([1001, 1001, 1001]) * units.m)


def test_strftime():
    """Test our monkey-patched xarray strftime."""
    data = xr.DataArray(np.datetime64('2000-01-01 01:00:00'))
    assert '2000-01-01 01:00:00' == data.dt.strftime('%Y-%m-%d %H:%M:%S')


def test_coordinates_basic_by_method(test_var):
    """Test that NARR example coordinates are like we expect using coordinates method."""
    x, y, vertical, time = test_var.metpy.coordinates('x', 'y', 'vertical', 'time')

    assert test_var['x'].identical(x)
    assert test_var['y'].identical(y)
    assert test_var['isobaric'].identical(vertical)
    assert test_var['time'].identical(time)


def test_coordinates_basic_by_property(test_var):
    """Test that NARR example coordinates are like we expect using properties."""
    assert test_var['x'].identical(test_var.metpy.x)
    assert test_var['y'].identical(test_var.metpy.y)
    assert test_var['isobaric'].identical(test_var.metpy.vertical)
    assert test_var['time'].identical(test_var.metpy.time)


def test_coordinates_specified_by_name_with_dataset(test_ds_generic):
    """Test that we can manually specify the coordinates by name."""
    data = test_ds_generic.metpy.parse_cf(coordinates={'T': 'a', 'Z': 'b', 'Y': 'c', 'X': 'd'})
    x, y, vertical, time = data['test'].metpy.coordinates('x', 'y', 'vertical', 'time')

    assert data['test']['d'].identical(x)
    assert data['test']['c'].identical(y)
    assert data['test']['b'].identical(vertical)
    assert data['test']['a'].identical(time)


def test_coordinates_specified_by_dataarray_with_dataset(test_ds_generic):
    """Test that we can manually specify the coordinates by DataArray."""
    data = test_ds_generic.metpy.parse_cf(coordinates={
        'T': test_ds_generic['d'],
        'Z': test_ds_generic['a'],
        'Y': test_ds_generic['b'],
        'X': test_ds_generic['c']
    })
    x, y, vertical, time = data['test'].metpy.coordinates('x', 'y', 'vertical', 'time')

    assert data['test']['c'].identical(x)
    assert data['test']['b'].identical(y)
    assert data['test']['a'].identical(vertical)
    assert data['test']['d'].identical(time)


def test_bad_coordinate_type(test_var):
    """Test that an AttributeError is raised when a bad axis/coordinate type is given."""
    with pytest.raises(AttributeError) as exc:
        next(test_var.metpy.coordinates('bad_axis_type'))
    assert 'not an interpretable axis' in str(exc.value)


def test_missing_coordinate_type(test_ds_generic):
    """Test that an AttributeError is raised when an axis/coordinate type is unavailable."""
    data = test_ds_generic.metpy.parse_cf('test', coordinates={'Z': 'e'})
    with pytest.raises(AttributeError) as exc:
        data.metpy.time
    assert 'not available' in str(exc.value)


def test_assign_axes_overwrite(test_ds_generic):
    """Test that CFConventionHandler._assign_axis overwrites past axis attributes."""
    data = test_ds_generic.copy()
    data['c'].attrs['axis'] = 'X'
    data.metpy._assign_axes({'Y': data['c']}, data['test'])
    assert data['c'].attrs['axis'] == 'Y'


def test_resolve_axis_conflict_lonlat_and_xy(test_ds_generic):
    """Test _resolve_axis_conflict with both lon/lat and x/y coordinates."""
    test_ds_generic['b'].attrs['_CoordinateAxisType'] = 'GeoX'
    test_ds_generic['c'].attrs['_CoordinateAxisType'] = 'Lon'
    test_ds_generic['d'].attrs['_CoordinateAxisType'] = 'GeoY'
    test_ds_generic['e'].attrs['_CoordinateAxisType'] = 'Lat'

    test_var = test_ds_generic.metpy.parse_cf('test')

    assert test_var['b'].identical(test_var.metpy.x)
    assert test_var['d'].identical(test_var.metpy.y)


def test_resolve_axis_conflict_double_lonlat(test_ds_generic):
    """Test _resolve_axis_conflict with double lon/lat coordinates."""
    test_ds_generic['b'].attrs['_CoordinateAxisType'] = 'Lat'
    test_ds_generic['c'].attrs['_CoordinateAxisType'] = 'Lon'
    test_ds_generic['d'].attrs['_CoordinateAxisType'] = 'Lat'
    test_ds_generic['e'].attrs['_CoordinateAxisType'] = 'Lon'

    with pytest.warns(UserWarning, match='Specify the unique'):
        test_ds_generic.metpy.parse_cf('test')


def test_resolve_axis_conflict_double_xy(test_ds_generic):
    """Test _resolve_axis_conflict with double x/y coordinates."""
    test_ds_generic['b'].attrs['standard_name'] = 'projection_x_coordinate'
    test_ds_generic['c'].attrs['standard_name'] = 'projection_y_coordinate'
    test_ds_generic['d'].attrs['standard_name'] = 'projection_x_coordinate'
    test_ds_generic['e'].attrs['standard_name'] = 'projection_y_coordinate'

    with pytest.warns(UserWarning, match='Specify the unique'):
        test_ds_generic.metpy.parse_cf('test')


def test_resolve_axis_conflict_double_x_with_single_dim(test_ds_generic):
    """Test _resolve_axis_conflict with double x coordinate, but only one being a dim."""
    test_ds_generic['e'].attrs['standard_name'] = 'projection_x_coordinate'
    test_ds_generic.coords['f'] = ('e', np.linspace(0, 1, 5))
    test_ds_generic['f'].attrs['standard_name'] = 'projection_x_coordinate'

    test_var = test_ds_generic.metpy.parse_cf('test')

    assert test_var['e'].identical(test_var.metpy.x)


def test_resolve_axis_conflict_double_vertical(test_ds_generic):
    """Test _resolve_axis_conflict with double vertical coordinates."""
    test_ds_generic['b'].attrs['units'] = 'hPa'
    test_ds_generic['c'].attrs['units'] = 'Pa'

    with pytest.warns(UserWarning, match='Specify the unique'):
        test_ds_generic.metpy.parse_cf('test')


criterion_matches = [
    ('standard_name', 'time', 'time'),
    ('standard_name', 'model_level_number', 'vertical'),
    ('standard_name', 'atmosphere_hybrid_sigma_pressure_coordinate', 'vertical'),
    ('standard_name', 'geopotential_height', 'vertical'),
    ('standard_name', 'height_above_geopotential_datum', 'vertical'),
    ('standard_name', 'altitude', 'vertical'),
    ('standard_name', 'atmosphere_sigma_coordinate', 'vertical'),
    ('standard_name', 'height_above_reference_ellipsoid', 'vertical'),
    ('standard_name', 'height', 'vertical'),
    ('standard_name', 'atmosphere_sleve_coordinate', 'vertical'),
    ('standard_name', 'height_above_mean_sea_level', 'vertical'),
    ('standard_name', 'atmosphere_hybrid_height_coordinate', 'vertical'),
    ('standard_name', 'atmosphere_ln_pressure_coordinate', 'vertical'),
    ('standard_name', 'air_pressure', 'vertical'),
    ('standard_name', 'projection_y_coordinate', 'y'),
    ('standard_name', 'latitude', 'lat'),
    ('standard_name', 'projection_x_coordinate', 'x'),
    ('standard_name', 'longitude', 'lon'),
    ('_CoordinateAxisType', 'Time', 'time'),
    ('_CoordinateAxisType', 'Pressure', 'vertical'),
    ('_CoordinateAxisType', 'GeoZ', 'vertical'),
    ('_CoordinateAxisType', 'Height', 'vertical'),
    ('_CoordinateAxisType', 'GeoY', 'y'),
    ('_CoordinateAxisType', 'Lat', 'lat'),
    ('_CoordinateAxisType', 'GeoX', 'x'),
    ('_CoordinateAxisType', 'Lon', 'lon'),
    ('axis', 'T', 'time'),
    ('axis', 'Z', 'vertical'),
    ('axis', 'Y', 'y'),
    ('axis', 'X', 'x'),
    ('positive', 'up', 'vertical'),
    ('positive', 'down', 'vertical')
]


@pytest.mark.parametrize('test_tuple', criterion_matches)
def test_check_axis_criterion_match(test_ds_generic, test_tuple):
    """Test the variety of possibilities for check_axis in the criterion match."""
    test_ds_generic['e'].attrs[test_tuple[0]] = test_tuple[1]
    assert test_ds_generic.metpy.check_axis(test_ds_generic['e'], test_tuple[2])


unit_matches = [
    ('Pa', 'vertical'),
    ('hPa', 'vertical'),
    ('mbar', 'vertical'),
    ('degreeN', 'lat'),
    ('degreesN', 'lat'),
    ('degree_north', 'lat'),
    ('degree_N', 'lat'),
    ('degrees_north', 'lat'),
    ('degrees_N', 'lat'),
    ('degreeE', 'lon'),
    ('degrees_east', 'lon'),
    ('degree_east', 'lon'),
    ('degreesE', 'lon'),
    ('degree_E', 'lon'),
    ('degrees_E', 'lon')
]


@pytest.mark.parametrize('test_tuple', unit_matches)
def test_check_axis_unit_match(test_ds_generic, test_tuple):
    """Test the variety of possibilities for check_axis in the unit match."""
    test_ds_generic['e'].attrs['units'] = test_tuple[0]
    assert test_ds_generic.metpy.check_axis(test_ds_generic['e'], test_tuple[1])


regex_matches = [
    ('time', 'time'),
    ('time1', 'time'),
    ('time42', 'time'),
    ('Time', 'time'),
    ('TIME', 'time'),
    ('bottom_top', 'vertical'),
    ('sigma', 'vertical'),
    ('HGHT', 'vertical'),
    ('height', 'vertical'),
    ('Altitude', 'vertical'),
    ('depth', 'vertical'),
    ('isobaric', 'vertical'),
    ('isobaric1', 'vertical'),
    ('isobaric42', 'vertical'),
    ('PRES', 'vertical'),
    ('pressure', 'vertical'),
    ('pressure_difference_layer', 'vertical'),
    ('isothermal', 'vertical'),
    ('y', 'y'),
    ('Y', 'y'),
    ('lat', 'lat'),
    ('latitude', 'lat'),
    ('Latitude', 'lat'),
    ('XLAT', 'lat'),
    ('x', 'x'),
    ('X', 'x'),
    ('lon', 'lon'),
    ('longitude', 'lon'),
    ('Longitude', 'lon'),
    ('XLONG', 'lon')
]


@pytest.mark.parametrize('test_tuple', regex_matches)
def test_check_axis_regular_expression_match(test_ds_generic, test_tuple):
    """Test the variety of possibilities for check_axis in the regular expression match."""
    data = test_ds_generic.rename({'e': test_tuple[0]})
    assert data.metpy.check_axis(data[test_tuple[0]], test_tuple[1])


def test_narr_example_variable_without_grid_mapping(test_ds):
    """Test that NARR example is parsed correctly, with x/y coordinates scaled the same."""
    data = test_ds.metpy.parse_cf()
    # Make sure that x and y coordinates are parsed correctly, rather than having unequal
    # scaling based on whether that variable has the grid_mapping attribute. This would
    # otherwise double the coordinates's shapes since xarray tries to combine the coordinates
    # with different scaling from differing units.
    assert test_ds['x'].shape == data['lon'].metpy.x.shape
    assert test_ds['y'].shape == data['lon'].metpy.y.shape
    assert data['lon'].metpy.x.identical(data['Temperature'].metpy.x)
    assert data['lon'].metpy.y.identical(data['Temperature'].metpy.y)


def test_coordinates_identical_true(test_ds_generic):
    """Test coordinates identical method when true."""
    assert test_ds_generic['test'].metpy.coordinates_identical(test_ds_generic['test'])


def test_coordinates_identical_false_number_of_coords(test_ds_generic):
    """Test coordinates identical method when false due to number of coordinates."""
    other_ds = test_ds_generic.drop('e')
    assert not test_ds_generic['test'].metpy.coordinates_identical(other_ds['test'])


def test_coordinates_identical_false_coords_mismatch(test_ds_generic):
    """Test coordinates identical method when false due to coordinates not matching."""
    other_ds = test_ds_generic.copy()
    other_ds['e'].attrs['units'] = 'meters'
    assert not test_ds_generic['test'].metpy.coordinates_identical(other_ds['test'])


def test_check_matching_coordinates(test_ds_generic):
    """Test xarray coordinate checking decorator."""
    other = test_ds_generic['test'].rename({'a': 'time'})

    @check_matching_coordinates
    def add(a, b):
        return a + b

    xr.testing.assert_identical(add(test_ds_generic['test'], test_ds_generic['test']),
                                test_ds_generic['test'] * 2)
    with pytest.raises(ValueError):
        add(test_ds_generic['test'], other)


def test_as_timestamp(test_var):
    """Test the as_timestamp method for a time DataArray."""
    time = test_var.metpy.time
    truth = xr.DataArray(np.array([544557600]),
                         name='time',
                         coords=time.coords,
                         dims='time',
                         attrs={'long_name': 'forecast time', 'axis': 'T',
                                'units': 'seconds'})
    assert truth.identical(time.metpy.as_timestamp())


def test_find_axis_name_integer(test_var):
    """Test getting axis name using the axis number identifier."""
    assert test_var.metpy.find_axis_name(2) == 'y'


def test_find_axis_name_axis_type(test_var):
    """Test getting axis name using the axis type identifier."""
    assert test_var.metpy.find_axis_name('vertical') == 'isobaric'


def test_find_axis_name_dim_coord_name(test_var):
    """Test getting axis name using the dimension coordinate name identifier."""
    assert test_var.metpy.find_axis_name('isobaric') == 'isobaric'


def test_find_axis_name_bad_identifier(test_var):
    """Test getting axis name using the axis type identifier."""
    with pytest.raises(ValueError) as exc:
        test_var.metpy.find_axis_name('latitude')
    assert 'axis is not valid' in str(exc.value)
