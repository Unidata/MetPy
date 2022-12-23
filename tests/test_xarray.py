# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the operation of MetPy's XArray accessors."""
from collections import OrderedDict
from unittest.mock import patch, PropertyMock

import numpy as np
import pyproj
import pytest
import xarray as xr

from metpy.plots.mapping import CFProjection
from metpy.testing import (assert_almost_equal, assert_array_almost_equal, assert_array_equal,
                           get_test_data)
from metpy.units import DimensionalityError, is_quantity, units
from metpy.xarray import (add_vertical_dim_from_xarray, check_axis, check_matching_coordinates,
                          grid_deltas_from_dataarray, preprocess_and_wrap)


@pytest.fixture
def test_ds():
    """Provide an xarray dataset for testing."""
    return xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))


@pytest.fixture
def test_ds_generic():
    """Provide a generic-coordinate dataset for testing."""
    return xr.DataArray(np.zeros((1, 3, 3, 5, 5)),
                        coords={
                            'a': xr.DataArray(np.arange(1), dims='a'),
                            'b': xr.DataArray(np.arange(3), dims='b'),
                            'c': xr.DataArray(np.arange(3), dims='c'),
                            'd': xr.DataArray(np.arange(5), dims='d'),
                            'e': xr.DataArray(np.arange(5), dims='e')
    }, dims=['a', 'b', 'c', 'd', 'e'], attrs={'units': 'kelvin'}, name='test').to_dataset()


@pytest.fixture
def test_var(test_ds):
    """Provide a standard, parsed, variable for testing."""
    return test_ds.metpy.parse_cf('Temperature')


@pytest.fixture
def test_var_multidim_full(test_ds):
    """Provide a variable with x/y coords and multidimensional lat/lon auxiliary coords."""
    return (test_ds[{'isobaric': [6, 12], 'y': [95, 96], 'x': [122, 123]}]
            .squeeze().set_coords(['lat', 'lon'])['Temperature'])


@pytest.fixture
def test_var_multidim_no_xy(test_var_multidim_full):
    """Provide a variable with multidimensional lat/lon coords but without x/y coords."""
    return test_var_multidim_full.drop_vars(['y', 'x'])


def test_projection(test_var, ccrs):
    """Test getting the proper projection out of the variable."""
    crs = test_var.metpy.crs

    assert crs['grid_mapping_name'] == 'lambert_conformal_conic'
    assert isinstance(test_var.metpy.cartopy_crs, ccrs.LambertConformal)


def test_pyproj_projection(test_var):
    """Test getting the proper pyproj projection out of the variable."""
    proj = test_var.metpy.pyproj_crs

    assert isinstance(proj, pyproj.CRS)
    assert proj.coordinate_operation.method_name == 'Lambert Conic Conformal (1SP)'


def test_no_projection(test_ds):
    """Test getting the crs attribute when not available produces a sensible error."""
    var = test_ds.lat
    with pytest.raises(AttributeError) as exc:
        var.metpy.crs

    assert 'not available' in str(exc.value)


def test_globe(test_var, ccrs):
    """Test getting the globe belonging to the projection."""
    globe = test_var.metpy.cartopy_globe

    assert globe.to_proj4_params() == OrderedDict([('ellps', 'sphere'),
                                                   ('a', 6367470.21484375),
                                                   ('b', 6367470.21484375)])
    assert isinstance(globe, ccrs.Globe)


def test_geodetic(test_var, ccrs):
    """Test getting the Geodetic CRS for the projection."""
    geodetic = test_var.metpy.cartopy_geodetic

    assert isinstance(geodetic, ccrs.Geodetic)


def test_unit_array(test_var):
    """Test unit handling through the accessor."""
    arr = test_var.metpy.unit_array
    assert is_quantity(arr)
    assert arr.units == units.kelvin


def test_units(test_var):
    """Test the units property on the accessor."""
    assert test_var.metpy.units == units.kelvin


def test_units_data(test_var):
    """Test units property fetching does not touch variable.data."""
    with patch.object(xr.Variable, 'data', new_callable=PropertyMock) as mock_data_property:
        test_var.metpy.units
        mock_data_property.assert_not_called()


def test_units_percent():
    """Test that '%' is handled as 'percent'."""
    test_var_percent = xr.open_dataset(
        get_test_data('irma_gfs_example.nc',
                      as_file_obj=False))['Relative_humidity_isobaric']
    assert test_var_percent.metpy.units == units.percent


def test_magnitude_with_quantity(test_var):
    """Test magnitude property on accessor when data is a quantity."""
    assert isinstance(test_var.metpy.magnitude, np.ndarray)
    np.testing.assert_array_almost_equal(test_var.metpy.magnitude, np.asarray(test_var.values))


def test_magnitude_without_quantity(test_ds_generic):
    """Test magnitude property on accessor when data is not a quantity."""
    assert isinstance(test_ds_generic['test'].data, np.ndarray)
    np.testing.assert_array_equal(
        test_ds_generic['test'].metpy.magnitude,
        np.asarray(test_ds_generic['test'].values)
    )


def test_convert_units(test_var):
    """Test conversion of units."""
    result = test_var.metpy.convert_units('degC')

    # Check that units are updated without modifying original
    assert result.metpy.units == units.degC
    assert test_var.metpy.units == units.kelvin

    # Make sure we now get an array back with properly converted values
    assert_almost_equal(result[0, 0, 0, 0], 18.44 * units.degC, 2)


def test_convert_to_base_units(test_ds):
    """Test conversion of units."""
    uwnd = test_ds.u_wind.metpy.quantify()
    result = (uwnd * (500 * units.hPa)).metpy.convert_to_base_units()

    # Check that units are updated without modifying original
    assert result.metpy.units == units('kg s**-3')
    assert test_ds.u_wind.metpy.units == units('m/s')

    # Make sure we now get an array back with properly converted values
    assert_almost_equal(result[0, 0, 0, 0], -448416.12 * units('kg s**-3'), 2)


def test_convert_coordinate_units(test_ds_generic):
    """Test conversion of coordinate units."""
    result = test_ds_generic['test'].metpy.convert_coordinate_units('b', 'percent')
    assert result['b'].data[1] == 100.
    assert result['b'].metpy.units == units.percent


def test_latlon_default_units(test_var_multidim_full):
    """Test that lat/lon are given degree units by default."""
    del test_var_multidim_full.lat.attrs['units']
    del test_var_multidim_full.lon.attrs['units']

    lat = test_var_multidim_full.metpy.latitude.metpy.unit_array
    assert lat.units == units.degrees
    assert lat.max() > 50 * units.degrees

    lon = test_var_multidim_full.metpy.longitude.metpy.unit_array
    assert lon.units == units.degrees
    assert lon.min() < -100 * units.degrees


def test_quantify(test_ds_generic):
    """Test quantify method for converting data to Quantity."""
    original = test_ds_generic['test'].values
    result = test_ds_generic['test'].metpy.quantify()
    assert is_quantity(result.data)
    assert result.data.units == units.kelvin
    assert 'units' not in result.attrs
    np.testing.assert_array_almost_equal(result.data, units.Quantity(original))


def test_dequantify():
    """Test dequantify method for converting data away from Quantity."""
    original = xr.DataArray(units.Quantity([280, 290, 300], 'K'),
                            attrs={'standard_name': 'air_temperature'})
    result = original.metpy.dequantify()
    assert isinstance(result.data, np.ndarray)
    assert result.attrs['units'] == 'kelvin'
    np.testing.assert_array_almost_equal(result.data, original.data.magnitude)
    assert result.attrs['standard_name'] == 'air_temperature'


def test_dataset_quantify(test_ds_generic):
    """Test quantify method for converting data to Quantity on Datasets."""
    result = test_ds_generic.metpy.quantify()
    assert is_quantity(result['test'].data)
    assert result['test'].data.units == units.kelvin
    assert 'units' not in result['test'].attrs
    np.testing.assert_array_almost_equal(
        result['test'].data,
        units.Quantity(test_ds_generic['test'].data)
    )
    assert result.attrs == test_ds_generic.attrs


def test_dataset_dequantify():
    """Test dequantify method for converting data away from Quantity on Datasets."""
    original = xr.Dataset({
        'test': ('x', units.Quantity([280, 290, 300], 'K')),
        'x': np.arange(3)
    }, attrs={'test': 'test'})
    result = original.metpy.dequantify()
    assert isinstance(result['test'].data, np.ndarray)
    assert result['test'].attrs['units'] == 'kelvin'
    np.testing.assert_array_almost_equal(result['test'].data, original['test'].data.magnitude)
    assert result.attrs == original.attrs


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


def test_missing_grid_mapping_valid():
    """Test falling back to implicit lat/lon projection when valid."""
    lon = xr.DataArray(-np.arange(3),
                       attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat = xr.DataArray(np.arange(2),
                       attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    data = xr.DataArray(np.arange(6).reshape(2, 3), coords=(lat, lon), dims=('y', 'x'))
    ds = xr.Dataset({'data': data})

    data_var = ds.metpy.parse_cf('data')
    assert (
        'metpy_crs' in data_var.coords
        and data_var.metpy.crs['grid_mapping_name'] == 'latitude_longitude'
    )


def test_missing_grid_mapping_invalid(test_var_multidim_no_xy):
    """Test not falling back to implicit lat/lon projection when invalid."""
    data_var = test_var_multidim_no_xy.to_dataset(name='data').metpy.parse_cf('data')
    assert 'metpy_crs' not in data_var.coords


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


def test_parsecf_crs():
    """Test calling `parse_cf` with the metpy_crs variable."""
    ds = xr.Dataset({'metpy_crs': xr.DataArray(1)})

    with pytest.warns(UserWarning, match='Attempting to parse metpy_crs'):
        ds.metpy.parse_cf('metpy_crs')


def test_parsecf_existing_scalar_crs():
    """Test calling `parse_cf` on a variable with an existing scalar metpy_crs coordinate."""
    ds = xr.Dataset({'data': xr.DataArray(1, coords={'metpy_crs': 1})})

    with pytest.warns(UserWarning, match='metpy_crs already present'):
        ds.metpy.parse_cf('data')


def test_parsecf_existing_vector_crs():
    """Test calling `parse_cf` on a variable with an existing vector metpy_crs coordinate."""
    ds = xr.Dataset({'data': xr.DataArray(1, dims=('metpy_crs',), coords=(np.ones(3),))})

    with pytest.warns(UserWarning, match='metpy_crs already present'):
        ds.metpy.parse_cf('data')


def test_preprocess_and_wrap_only_preprocessing():
    """Test xarray preprocessing and wrapping decorator for only preprocessing."""
    data = xr.DataArray(np.ones(3), attrs={'units': 'km'})
    data2 = xr.DataArray(np.ones(3), attrs={'units': 'm'})

    @preprocess_and_wrap()
    def func(a, b):
        return a.to('m') + b

    assert_array_equal(func(data, b=data2), np.array([1001, 1001, 1001]) * units.m)


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
    data = test_ds_generic.metpy.parse_cf(coordinates={'time': 'a', 'vertical': 'b', 'y': 'c',
                                                       'x': 'd'})
    x, y, vertical, time = data['test'].metpy.coordinates('x', 'y', 'vertical', 'time')

    assert data['test']['d'].identical(x)
    assert data['test']['c'].identical(y)
    assert data['test']['b'].identical(vertical)
    assert data['test']['a'].identical(time)


def test_coordinates_specified_by_dataarray_with_dataset(test_ds_generic):
    """Test that we can manually specify the coordinates by DataArray."""
    data = test_ds_generic.metpy.parse_cf(coordinates={
        'time': test_ds_generic['d'],
        'vertical': test_ds_generic['a'],
        'y': test_ds_generic['b'],
        'x': test_ds_generic['c']
    })
    x, y, vertical, time = data['test'].metpy.coordinates('x', 'y', 'vertical', 'time')

    assert data['test']['c'].identical(x)
    assert data['test']['b'].identical(y)
    assert data['test']['a'].identical(vertical)
    assert data['test']['d'].identical(time)


def test_missing_coordinate_type(test_ds_generic):
    """Test that an AttributeError is raised when an axis/coordinate type is unavailable."""
    data = test_ds_generic.metpy.parse_cf('test', coordinates={'vertical': 'e'})
    with pytest.raises(AttributeError) as exc:
        data.metpy.time
    assert 'not available' in str(exc.value)


def test_assign_coordinates_not_overwrite(test_ds_generic):
    """Test that assign_coordinates does not overwrite past axis attributes."""
    data = test_ds_generic.copy()
    data['c'].attrs['axis'] = 'X'
    data['test'] = data['test'].metpy.assign_coordinates({'y': data['c']})
    assert data['c'].identical(data['test'].metpy.y)
    assert data['c'].attrs['axis'] == 'X'


def test_resolve_axis_conflict_lonlat_and_xy(test_ds_generic):
    """Test _resolve_axis_conflict with both lon/lat and x/y coordinates."""
    test_ds_generic['b'].attrs['_CoordinateAxisType'] = 'GeoX'
    test_ds_generic['c'].attrs['_CoordinateAxisType'] = 'Lon'
    test_ds_generic['d'].attrs['_CoordinateAxisType'] = 'GeoY'
    test_ds_generic['e'].attrs['_CoordinateAxisType'] = 'Lat'

    assert test_ds_generic['test'].metpy.x.name == 'b'
    assert test_ds_generic['test'].metpy.y.name == 'd'


def test_resolve_axis_conflict_double_lonlat(test_ds_generic):
    """Test _resolve_axis_conflict with double lon/lat coordinates."""
    test_ds_generic['b'].attrs['_CoordinateAxisType'] = 'Lat'
    test_ds_generic['c'].attrs['_CoordinateAxisType'] = 'Lon'
    test_ds_generic['d'].attrs['_CoordinateAxisType'] = 'Lat'
    test_ds_generic['e'].attrs['_CoordinateAxisType'] = 'Lon'

    with pytest.warns(UserWarning, match='More than one x coordinate'),\
            pytest.raises(AttributeError):
        test_ds_generic['test'].metpy.x
    with pytest.warns(UserWarning, match='More than one y coordinate'),\
            pytest.raises(AttributeError):
        test_ds_generic['test'].metpy.y


def test_resolve_axis_conflict_double_xy(test_ds_generic):
    """Test _resolve_axis_conflict with double x/y coordinates."""
    test_ds_generic['b'].attrs['standard_name'] = 'projection_x_coordinate'
    test_ds_generic['c'].attrs['standard_name'] = 'projection_y_coordinate'
    test_ds_generic['d'].attrs['standard_name'] = 'projection_x_coordinate'
    test_ds_generic['e'].attrs['standard_name'] = 'projection_y_coordinate'

    with pytest.warns(UserWarning, match='More than one x coordinate'),\
            pytest.raises(AttributeError):
        test_ds_generic['test'].metpy.x
    with pytest.warns(UserWarning, match='More than one y coordinate'),\
            pytest.raises(AttributeError):
        test_ds_generic['test'].metpy.y


def test_resolve_axis_conflict_double_x_with_single_dim(test_ds_generic):
    """Test _resolve_axis_conflict with double x coordinate, but only one being a dim."""
    test_ds_generic['e'].attrs['standard_name'] = 'projection_x_coordinate'
    test_ds_generic.coords['f'] = ('e', np.linspace(0, 1, 5))
    test_ds_generic['f'].attrs['standard_name'] = 'projection_x_coordinate'

    assert test_ds_generic['test'].metpy.x.name == 'e'


def test_resolve_axis_conflict_double_vertical(test_ds_generic):
    """Test _resolve_axis_conflict with double vertical coordinates."""
    test_ds_generic['b'].attrs['units'] = 'hPa'
    test_ds_generic['c'].attrs['units'] = 'Pa'

    with pytest.warns(UserWarning, match='More than one vertical coordinate'),\
            pytest.raises(AttributeError):
        test_ds_generic['test'].metpy.vertical


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
    ('standard_name', 'latitude', 'latitude'),
    ('standard_name', 'projection_x_coordinate', 'x'),
    ('standard_name', 'longitude', 'longitude'),
    ('_CoordinateAxisType', 'Time', 'time'),
    ('_CoordinateAxisType', 'Pressure', 'vertical'),
    ('_CoordinateAxisType', 'GeoZ', 'vertical'),
    ('_CoordinateAxisType', 'Height', 'vertical'),
    ('_CoordinateAxisType', 'GeoY', 'y'),
    ('_CoordinateAxisType', 'Lat', 'latitude'),
    ('_CoordinateAxisType', 'GeoX', 'x'),
    ('_CoordinateAxisType', 'Lon', 'longitude'),
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
    assert check_axis(test_ds_generic['e'], test_tuple[2])


unit_matches = [
    ('Pa', 'vertical'),
    ('hPa', 'vertical'),
    ('mbar', 'vertical'),
    ('degreeN', 'latitude'),
    ('degreesN', 'latitude'),
    ('degree_north', 'latitude'),
    ('degree_N', 'latitude'),
    ('degrees_north', 'latitude'),
    ('degrees_N', 'latitude'),
    ('degreeE', 'longitude'),
    ('degrees_east', 'longitude'),
    ('degree_east', 'longitude'),
    ('degreesE', 'longitude'),
    ('degree_E', 'longitude'),
    ('degrees_E', 'longitude')
]


@pytest.mark.parametrize('test_tuple', unit_matches)
def test_check_axis_unit_match(test_ds_generic, test_tuple):
    """Test the variety of possibilities for check_axis in the unit match."""
    test_ds_generic['e'].attrs['units'] = test_tuple[0]
    assert check_axis(test_ds_generic['e'], test_tuple[1])


regex_matches = [
    ('time', 'time'),
    ('time1', 'time'),
    ('time42', 'time'),
    ('Time', 'time'),
    ('TIME', 'time'),
    ('XTIME', 'time'),
    ('Times', 'time'),
    ('bottom_top', 'vertical'),
    ('bottom_top_stag', 'vertical'),
    ('sigma', 'vertical'),
    ('HGHT', 'vertical'),
    ('height', 'vertical'),
    ('Altitude', 'vertical'),
    ('depth', 'vertical'),
    ('isobaric', 'vertical'),
    ('isobaric1', 'vertical'),
    ('isobaric42', 'vertical'),
    ('lv_HTGL5', 'vertical'),
    ('PRES', 'vertical'),
    ('pressure', 'vertical'),
    ('pressure_difference_layer', 'vertical'),
    ('isothermal', 'vertical'),
    ('z', 'vertical'),
    ('z_stag', 'vertical'),
    ('y', 'y'),
    ('Y', 'y'),
    ('y_stag', 'y'),
    ('yc', 'y'),
    ('lat', 'latitude'),
    ('latitude', 'latitude'),
    ('Latitude', 'latitude'),
    ('XLAT', 'latitude'),
    ('XLAT_U', 'latitude'),
    ('x', 'x'),
    ('X', 'x'),
    ('x_stag', 'x'),
    ('xc', 'x'),
    ('lon', 'longitude'),
    ('longitude', 'longitude'),
    ('Longitude', 'longitude'),
    ('XLONG', 'longitude'),
    ('XLONG_V', 'longitude')
]


@pytest.mark.parametrize('test_tuple', regex_matches)
def test_check_axis_regular_expression_match(test_ds_generic, test_tuple):
    """Test the variety of possibilities for check_axis in the regular expression match."""
    data = test_ds_generic.rename({'e': test_tuple[0]})
    assert check_axis(data[test_tuple[0]], test_tuple[1])


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
    other_ds = test_ds_generic.drop_vars('e')
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


def test_time_deltas():
    """Test the time_deltas attribute."""
    ds = xr.open_dataset(get_test_data('irma_gfs_example.nc', as_file_obj=False))
    time = ds['time1']
    truth = 3 * np.ones(8) * units.hr
    assert_array_almost_equal(time.metpy.time_deltas, truth)


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
        test_var.metpy.find_axis_name('ens')
    assert 'axis is not valid' in str(exc.value)


def test_find_axis_number_integer(test_var):
    """Test getting axis number using the axis number identifier."""
    assert test_var.metpy.find_axis_number(2) == 2


def test_find_axis_number_axis_type(test_var):
    """Test getting axis number using the axis type identifier."""
    assert test_var.metpy.find_axis_number('vertical') == 1


def test_find_axis_number_dim_coord_number(test_var):
    """Test getting axis number using the dimension coordinate name identifier."""
    assert test_var.metpy.find_axis_number('isobaric') == 1


def test_find_axis_number_bad_identifier(test_var):
    """Test getting axis number using the axis type identifier."""
    with pytest.raises(ValueError) as exc:
        test_var.metpy.find_axis_number('ens')
    assert 'axis is not valid' in str(exc.value)


def test_cf_parse_with_grid_mapping(test_var):
    """Test cf_parse dont delete grid_mapping attribute."""
    assert test_var.grid_mapping == 'Lambert_Conformal'


def test_data_array_loc_get_with_units(test_var):
    """Test the .loc indexer on the metpy accessor."""
    truth = test_var.loc[:, 850.]
    assert truth.identical(test_var.metpy.loc[:, 8.5e4 * units.Pa])


def test_data_array_loc_set_with_units(test_var):
    """Test the .loc indexer on the metpy accessor for setting."""
    temperature = test_var.copy()
    temperature.metpy.loc[:, 8.5e4 * units.Pa] = np.nan
    assert np.isnan(temperature.loc[:, 850.]).all()
    assert not np.isnan(temperature.loc[:, 700.]).any()


def test_data_array_loc_with_ellipsis(test_var):
    """Test the .loc indexer using multiple Ellipses to verify expansion behavior."""
    truth = test_var[:, :, -1, :]
    assert truth.identical(test_var.metpy.loc[..., 711.3653535503963 * units.km, ...])


def test_data_array_loc_non_tuple(test_var):
    """Test the .loc indexer with a non-tuple indexer."""
    truth = test_var[-1]
    assert truth.identical(test_var.metpy.loc['1987-04-04T18:00'])


def test_data_array_loc_too_many_indices(test_var):
    """Test the .loc indexer when too many indices are given."""
    with pytest.raises(IndexError):
        test_var.metpy.loc[:, 8.5e4 * units.Pa, :, :, :]


def test_data_array_sel_dict_with_units(test_var):
    """Test .sel on the metpy accessor with dictionary."""
    truth = test_var.squeeze().loc[500.]
    assert truth.identical(test_var.metpy.sel({'time': '1987-04-04T18:00:00',
                                               'isobaric': 5e4 * units.Pa}))


def test_data_array_sel_kwargs_with_units(test_var):
    """Test .sel on the metpy accessor with kwargs and axis type."""
    truth = test_var.loc[:, 500.][..., 122]
    selection = (
        test_var.metpy
        .sel(vertical=5e4 * units.Pa, x=-16.569 * units.km, tolerance=1., method='nearest')
        .metpy
        .assign_coordinates(None)
    )
    assert truth.identical(selection)


def test_dataset_loc_with_units(test_ds):
    """Test .loc on the metpy accessor for Datasets using slices."""
    truth = test_ds[{'isobaric': slice(6, 17)}]
    assert truth.identical(test_ds.metpy.loc[{'isobaric': slice(8.5e4 * units.Pa,
                                                                5e4 * units.Pa)}])


def test_dataset_sel_kwargs_with_units(test_ds):
    """Test .sel on the metpy accessor for Datasets with kwargs."""
    truth = test_ds[{'time': 0, 'y': 50, 'x': 122}]
    assert truth.identical(test_ds.metpy.sel(time='1987-04-04T18:00:00', y=-1.464e6 * units.m,
                                             x=-17. * units.km, tolerance=1.,
                                             method='nearest'))


def test_dataset_sel_non_dict_pos_arg(test_ds):
    """Test that .sel errors when first positional argument is not a dict."""
    with pytest.raises(ValueError) as exc:
        test_ds.metpy.sel('1987-04-04T18:00:00')
    assert 'must be a dictionary' in str(exc.value)


def test_dataset_sel_mixed_dict_and_kwarg(test_ds):
    """Test that .sel errors when dict positional argument and kwargs are mixed."""
    with pytest.raises(ValueError) as exc:
        test_ds.metpy.sel({'isobaric': slice(8.5e4 * units.Pa, 5e4 * units.Pa)},
                          time='1987-04-04T18:00:00')
    assert 'cannot specify both keyword and positional arguments' in str(exc.value)


def test_dataset_loc_without_dict(test_ds):
    """Test that .metpy.loc for Datasets raises error when used with a non-dict."""
    with pytest.raises(TypeError):
        test_ds.metpy.loc[:, 700 * units.hPa]


def test_dataset_parse_cf_keep_attrs(test_ds):
    """Test that .parse_cf() does not remove attributes on the parsed dataset."""
    parsed_ds = test_ds.metpy.parse_cf()

    assert parsed_ds.attrs  # Must be non-empty
    assert parsed_ds.attrs == test_ds.attrs  # Must match


def test_check_axis_with_bad_unit(test_ds_generic):
    """Test that check_axis does not raise an exception when provided a bad unit."""
    var = test_ds_generic['e']
    var.attrs['units'] = 'nondimensional'
    assert not check_axis(var, 'x', 'y', 'vertical', 'time')


def test_dataset_parse_cf_varname_list(test_ds):
    """Test that .parse_cf() returns correct subset of dataset when given list of vars."""
    full_ds = test_ds.copy().metpy.parse_cf()
    partial_ds = test_ds.metpy.parse_cf(['u_wind', 'v_wind'])

    assert full_ds[['u_wind', 'v_wind']].identical(partial_ds)


def test_coordinate_identification_shared_but_not_equal_coords():
    """Test that non-shared coords are not skipped after parsing shared coords.

    See GH Issue #1124.
    """
    # Create minimal dataset
    temperature = xr.DataArray([[8, 4], [13, 12]], name='temperature',
                               dims=('isobaric1', 'x'),
                               coords={
                                   'isobaric1': xr.DataArray([700, 850],
                                                             name='isobaric1',
                                                             dims='isobaric1',
                                                             attrs={'units': 'hPa',
                                                                    'axis': 'Z'}),
                                   'x': xr.DataArray([0, 450], name='x', dims='x',
                                                     attrs={'units': 'km', 'axis': 'X'})},
                               attrs={'units': 'degC'})
    u = xr.DataArray([[30, 20], [10, 10]], name='u', dims=('isobaric2', 'x'),
                     coords={
                         'isobaric2': xr.DataArray([500, 850], name='isobaric2',
                                                   dims='isobaric2',
                                                   attrs={'units': 'hPa', 'axis': 'Z'}),
                         'x': xr.DataArray([0, 450], name='x', dims='x',
                                           attrs={'units': 'km', 'axis': 'X'})},
                     attrs={'units': 'kts'})
    ds = xr.Dataset({'temperature': temperature, 'u': u})

    # Check coordinates on temperature
    assert ds['isobaric1'].identical(ds['temperature'].metpy.vertical)
    assert ds['x'].identical(ds['temperature'].metpy.x)

    # Check vertical coordinate on u
    # Fails prior to resolution of Issue #1124
    assert ds['isobaric2'].identical(ds['u'].metpy.vertical)


def test_one_dimensional_lat_lon(test_ds_generic):
    """Test that 1D lat/lon coords are recognized as both x/y and longitude/latitude."""
    test_ds_generic['d'].attrs['units'] = 'degrees_north'
    test_ds_generic['e'].attrs['units'] = 'degrees_east'
    var = test_ds_generic.metpy.parse_cf('test')
    assert var['d'].identical(var.metpy.y)
    assert var['d'].identical(var.metpy.latitude)
    assert var['e'].identical(var.metpy.x)
    assert var['e'].identical(var.metpy.longitude)


def test_auxilary_lat_lon_with_xy(test_var_multidim_full):
    """Test that auxiliary lat/lon coord identification works with other x/y coords present."""
    assert test_var_multidim_full['y'].identical(test_var_multidim_full.metpy.y)
    assert test_var_multidim_full['lat'].identical(test_var_multidim_full.metpy.latitude)
    assert test_var_multidim_full['x'].identical(test_var_multidim_full.metpy.x)
    assert test_var_multidim_full['lon'].identical(test_var_multidim_full.metpy.longitude)


def test_auxilary_lat_lon_without_xy(test_var_multidim_no_xy):
    """Test that multidimensional lat/lon are recognized in absence of x/y coords."""
    assert test_var_multidim_no_xy['lat'].identical(test_var_multidim_no_xy.metpy.latitude)
    assert test_var_multidim_no_xy['lon'].identical(test_var_multidim_no_xy.metpy.longitude)


def test_auxilary_lat_lon_without_xy_as_xy(test_var_multidim_no_xy):
    """Test that the pre-v1.0 behavior of multidimensional lat/lon errors."""
    with pytest.raises(AttributeError):
        test_var_multidim_no_xy.metpy.y

    with pytest.raises(AttributeError):
        test_var_multidim_no_xy.metpy.x


# Declare a sample projection with CF attributes
sample_cf_attrs = {
    'grid_mapping_name': 'lambert_conformal_conic',
    'earth_radius': 6370000,
    'standard_parallel': [30., 40.],
    'longitude_of_central_meridian': 260.,
    'latitude_of_projection_origin': 35.
}


def test_assign_crs_dataarray_by_argument(test_ds_generic, ccrs):
    """Test assigning CRS to DataArray by projection dict."""
    da = test_ds_generic['test']
    new_da = da.metpy.assign_crs(sample_cf_attrs)
    assert isinstance(new_da.metpy.cartopy_crs, ccrs.LambertConformal)
    assert new_da['metpy_crs'] == CFProjection(sample_cf_attrs)


def test_assign_crs_dataarray_by_kwargs(test_ds_generic, ccrs):
    """Test assigning CRS to DataArray by projection kwargs."""
    da = test_ds_generic['test']
    new_da = da.metpy.assign_crs(**sample_cf_attrs)
    assert isinstance(new_da.metpy.cartopy_crs, ccrs.LambertConformal)
    assert new_da['metpy_crs'] == CFProjection(sample_cf_attrs)


def test_assign_crs_dataset_by_argument(test_ds_generic, ccrs):
    """Test assigning CRS to Dataset by projection dict."""
    new_ds = test_ds_generic.metpy.assign_crs(sample_cf_attrs)
    assert isinstance(new_ds['test'].metpy.cartopy_crs, ccrs.LambertConformal)
    assert new_ds['metpy_crs'] == CFProjection(sample_cf_attrs)


def test_assign_crs_dataset_by_kwargs(test_ds_generic, ccrs):
    """Test assigning CRS to Dataset by projection kwargs."""
    new_ds = test_ds_generic.metpy.assign_crs(**sample_cf_attrs)
    assert isinstance(new_ds['test'].metpy.cartopy_crs, ccrs.LambertConformal)
    assert new_ds['metpy_crs'] == CFProjection(sample_cf_attrs)


def test_assign_crs_error_with_both_attrs(test_ds_generic):
    """Test ValueError is raised when both dictionary and kwargs given."""
    with pytest.raises(ValueError) as exc:
        test_ds_generic.metpy.assign_crs(sample_cf_attrs, **sample_cf_attrs)
    assert 'Cannot specify both' in str(exc)


def test_assign_crs_error_with_neither_attrs(test_ds_generic):
    """Test ValueError is raised when neither dictionary and kwargs given."""
    with pytest.raises(ValueError) as exc:
        test_ds_generic.metpy.assign_crs()
    assert 'Must specify either' in str(exc)


def test_assign_latitude_longitude_no_horizontal(test_ds_generic):
    """Test that assign_latitude_longitude only warns when no horizontal coordinates."""
    with pytest.warns(UserWarning):
        xr.testing.assert_identical(test_ds_generic,
                                    test_ds_generic.metpy.assign_latitude_longitude())


def test_assign_y_x_no_horizontal(test_ds_generic):
    """Test that assign_y_x only warns when no horizontal coordinates."""
    with pytest.warns(UserWarning):
        xr.testing.assert_identical(test_ds_generic,
                                    test_ds_generic.metpy.assign_y_x())


@pytest.fixture
def test_coord_helper_da_yx():
    """Provide a DataArray with y/x coords for coord helpers."""
    return xr.DataArray(np.arange(9).reshape((3, 3)),
                        dims=('y', 'x'),
                        coords={'y': np.linspace(0, 1e5, 3),
                                'x': np.linspace(-1e5, 0, 3),
                                'metpy_crs': CFProjection(sample_cf_attrs)})


@pytest.fixture
def test_coord_helper_da_dummy_latlon(test_coord_helper_da_yx):
    """Provide DataArray with bad dummy lat/lon coords to be overwritten."""
    return test_coord_helper_da_yx.assign_coords(latitude=0., longitude=0.)


@pytest.fixture
def test_coord_helper_da_latlon():
    """Provide a DataArray with lat/lon coords for coord helpers."""
    return xr.DataArray(
        np.arange(9).reshape((3, 3)),
        dims=('y', 'x'),
        coords={
            'latitude': xr.DataArray(
                np.array(
                    [[34.99501239, 34.99875307, 35.],
                     [35.44643155, 35.45019292, 35.45144675],
                     [35.89782579, 35.90160784, 35.90286857]]
                ),
                dims=('y', 'x')
            ),
            'longitude': xr.DataArray(
                np.array(
                    [[-101.10219213, -100.55111288, -100.],
                     [-101.10831414, -100.55417417, -100.],
                     [-101.11450453, -100.55726965, -100.]]
                ),
                dims=('y', 'x')
            ),
            'metpy_crs': CFProjection(sample_cf_attrs)
        }
    )


@pytest.fixture
def test_coord_helper_da_dummy_yx(test_coord_helper_da_latlon):
    """Provide DataArray with bad dummy y/x coords to be overwritten."""
    return test_coord_helper_da_latlon.assign_coords(y=range(3), x=range(3))


def test_assign_latitude_longitude_basic_dataarray(test_coord_helper_da_yx,
                                                   test_coord_helper_da_latlon):
    """Test assign_latitude_longitude in basic usage on DataArray."""
    new_da = test_coord_helper_da_yx.metpy.assign_latitude_longitude()
    lat, lon = new_da.metpy.coordinates('latitude', 'longitude')
    np.testing.assert_array_almost_equal(test_coord_helper_da_latlon['latitude'].values,
                                         lat.values, 3)
    np.testing.assert_array_almost_equal(test_coord_helper_da_latlon['longitude'].values,
                                         lon.values, 3)


def test_assign_latitude_longitude_error_existing_dataarray(
        test_coord_helper_da_dummy_latlon):
    """Test assign_latitude_longitude failure with existing coordinates."""
    with pytest.raises(RuntimeError) as exc:
        test_coord_helper_da_dummy_latlon.metpy.assign_latitude_longitude()
    assert 'Latitude/longitude coordinate(s) are present' in str(exc)


def test_assign_latitude_longitude_force_existing_dataarray(
        test_coord_helper_da_dummy_latlon, test_coord_helper_da_latlon):
    """Test assign_latitude_longitude with existing coordinates forcing new."""
    new_da = test_coord_helper_da_dummy_latlon.metpy.assign_latitude_longitude(True)
    lat, lon = new_da.metpy.coordinates('latitude', 'longitude')
    np.testing.assert_array_almost_equal(test_coord_helper_da_latlon['latitude'].values,
                                         lat.values, 3)
    np.testing.assert_array_almost_equal(test_coord_helper_da_latlon['longitude'].values,
                                         lon.values, 3)


def test_assign_latitude_longitude_basic_dataset(test_coord_helper_da_yx,
                                                 test_coord_helper_da_latlon):
    """Test assign_latitude_longitude in basic usage on Dataset."""
    ds = test_coord_helper_da_yx.to_dataset(name='test').metpy.assign_latitude_longitude()
    lat, lon = ds['test'].metpy.coordinates('latitude', 'longitude')
    np.testing.assert_array_almost_equal(test_coord_helper_da_latlon['latitude'].values,
                                         lat.values, 3)
    np.testing.assert_array_almost_equal(test_coord_helper_da_latlon['longitude'].values,
                                         lon.values, 3)


def test_assign_y_x_basic_dataarray(test_coord_helper_da_yx, test_coord_helper_da_latlon):
    """Test assign_y_x in basic usage on DataArray."""
    new_da = test_coord_helper_da_latlon.metpy.assign_y_x()
    y, x = new_da.metpy.coordinates('y', 'x')
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['y'].values, y.values, 3)
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['x'].values, x.values, 3)


def test_assign_y_x_error_existing_dataarray(
        test_coord_helper_da_dummy_yx):
    """Test assign_y_x failure with existing coordinates."""
    with pytest.raises(RuntimeError) as exc:
        test_coord_helper_da_dummy_yx.metpy.assign_y_x()
    assert 'y/x coordinate(s) are present' in str(exc)


def test_assign_y_x_force_existing_dataarray(
        test_coord_helper_da_dummy_yx, test_coord_helper_da_yx):
    """Test assign_y_x with existing coordinates forcing new."""
    new_da = test_coord_helper_da_dummy_yx.metpy.assign_y_x(True)
    y, x = new_da.metpy.coordinates('y', 'x')
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['y'].values, y.values, 3)
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['x'].values, x.values, 3)


def test_assign_y_x_dataarray_outside_tolerance(test_coord_helper_da_latlon):
    """Test assign_y_x raises ValueError when tolerance is exceeded on DataArray."""
    with pytest.raises(ValueError) as exc:
        test_coord_helper_da_latlon.metpy.assign_y_x(tolerance=1 * units('um'))
    assert 'cannot be collapsed to 1D within tolerance' in str(exc)


def test_assign_y_x_dataarray_transposed(test_coord_helper_da_yx, test_coord_helper_da_latlon):
    """Test assign_y_x on DataArray with transposed order."""
    new_da = test_coord_helper_da_latlon.transpose(transpose_coords=True).metpy.assign_y_x()
    y, x = new_da.metpy.coordinates('y', 'x')
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['y'].values, y.values, 3)
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['x'].values, x.values, 3)


def test_assign_y_x_dataset_assumed_order(test_coord_helper_da_yx,
                                          test_coord_helper_da_latlon):
    """Test assign_y_x on Dataset where order must be assumed."""
    with pytest.warns(UserWarning):
        new_ds = test_coord_helper_da_latlon.to_dataset(name='test').rename_dims(
            {'y': 'b', 'x': 'a'}).metpy.assign_y_x()
    y, x = new_ds['test'].metpy.coordinates('y', 'x')
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['y'].values, y.values, 3)
    np.testing.assert_array_almost_equal(test_coord_helper_da_yx['x'].values, x.values, 3)
    assert y.name == 'b'
    assert x.name == 'a'


def test_assign_y_x_error_existing_dataset(
        test_coord_helper_da_dummy_yx):
    """Test assign_y_x failure with existing coordinates for Dataset."""
    with pytest.raises(RuntimeError) as exc:
        test_coord_helper_da_dummy_yx.to_dataset(name='test').metpy.assign_y_x()
    assert 'y/x coordinate(s) are present' in str(exc)


def test_update_attribute_dictionary(test_ds_generic):
    """Test update_attribute using dictionary."""
    descriptions = {
        'test': 'Filler data',
        'c': 'The third coordinate'
    }
    test_ds_generic.c.attrs['units'] = 'K'
    test_ds_generic.a.attrs['standard_name'] = 'air_temperature'
    result = test_ds_generic.metpy.update_attribute('description', descriptions)

    # Test attribute updates
    assert 'description' not in result['a'].attrs
    assert 'description' not in result['b'].attrs
    assert result['c'].attrs['description'] == 'The third coordinate'
    assert 'description' not in result['d'].attrs
    assert 'description' not in result['e'].attrs
    assert result['test'].attrs['description'] == 'Filler data'

    # Test that other attributes remain
    assert result['c'].attrs['units'] == 'K'
    assert result['a'].attrs['standard_name'] == 'air_temperature'

    # Test for no side effects
    assert 'description' not in test_ds_generic['c'].attrs
    assert 'description' not in test_ds_generic['test'].attrs


def test_update_attribute_callable(test_ds_generic):
    """Test update_attribute using callable."""
    def even_ascii(varname, **kwargs):
        return 'yes' if ord(varname[0]) % 2 == 0 else None

    result = test_ds_generic.metpy.update_attribute('even', even_ascii)

    # Test attribute updates
    assert 'even' not in result['a'].attrs
    assert result['b'].attrs['even'] == 'yes'
    assert 'even' not in result['c'].attrs
    assert result['d'].attrs['even'] == 'yes'
    assert 'even' not in result['e'].attrs
    assert result['test'].attrs['even'] == 'yes'

    # Test for no side effects
    assert 'even' not in test_ds_generic['b'].attrs
    assert 'even' not in test_ds_generic['d'].attrs
    assert 'even' not in test_ds_generic['test'].attrs
    test_ds_generic.metpy.update_attribute('even', even_ascii)


@pytest.mark.parametrize('test, other, match_unit, expected', [
    (np.arange(4), np.arange(4), False, np.arange(4)),
    (np.arange(4), np.arange(4), True, np.arange(4) * units('dimensionless')),
    (np.arange(4), [0] * units.m, False, np.arange(4) * units('dimensionless')),
    (np.arange(4), [0] * units.m, True, np.arange(4) * units.m),
    (
        np.arange(4),
        xr.DataArray(
            np.zeros(4) * units.meter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 4)},
            attrs={'description': 'Just some zeros'}
        ),
        False,
        xr.DataArray(
            np.arange(4) * units.dimensionless,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 4)}
        )
    ),
    (
        np.arange(4),
        xr.DataArray(
            np.zeros(4) * units.meter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 4)},
            attrs={'description': 'Just some zeros'}
        ),
        True,
        xr.DataArray(
            np.arange(4) * units.meter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 4)}
        )
    ),
    ([2, 4, 8] * units.kg, [0] * units.m, False, [2, 4, 8] * units.kg),
    ([2, 4, 8] * units.kg, [0] * units.g, True, [2000, 4000, 8000] * units.g),
    (
        [2, 4, 8] * units.kg,
        xr.DataArray(
            np.zeros(3) * units.meter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 3)}
        ),
        False,
        xr.DataArray(
            [2, 4, 8] * units.kilogram,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 3)}
        )
    ),
    (
        [2, 4, 8] * units.kg,
        xr.DataArray(
            np.zeros(3) * units.gram,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 3)}
        ),
        True,
        xr.DataArray(
            [2000, 4000, 8000] * units.gram,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 3)}
        )
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter,
            attrs={'description': 'A range of values'}
        ),
        np.arange(4, dtype=np.float64),
        False,
        units.Quantity(np.linspace(0, 1, 5), 'meter')
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter,
            attrs={'description': 'A range of values'}
        ),
        [0] * units.kg,
        False,
        np.linspace(0, 1, 5) * units.m
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter,
            attrs={'description': 'A range of values'}
        ),
        [0] * units.cm,
        True,
        np.linspace(0, 100, 5) * units.cm
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter,
            attrs={'description': 'A range of values'}
        ),
        xr.DataArray(
            np.zeros(5) * units.kilogram,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 5)},
            attrs={'description': 'Alternative data'}
        ),
        False,
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 5)}
        )
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter,
            attrs={'description': 'A range of values'}
        ),
        xr.DataArray(
            np.zeros(5) * units.centimeter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 5)},
            attrs={'description': 'Alternative data'}
        ),
        True,
        xr.DataArray(
            np.linspace(0, 100, 5) * units.centimeter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 5)}
        )
    ),
])
def test_wrap_with_wrap_like_kwarg(test, other, match_unit, expected):
    """Test the preprocess and wrap decorator when using wrap_like."""
    @preprocess_and_wrap(wrap_like=other, match_unit=match_unit)
    def almost_identity(arg):
        return arg

    result = almost_identity(test)

    if hasattr(expected, 'units'):
        assert expected.units == result.units
    if isinstance(expected, xr.DataArray):
        xr.testing.assert_identical(result, expected)
    else:
        assert_array_equal(result, expected)


@pytest.mark.parametrize('test, other', [
    ([2, 4, 8] * units.kg, [0] * units.m),
    (
        [2, 4, 8] * units.kg,
        xr.DataArray(
            np.zeros(3) * units.meter,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 3)}
        )
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter
        ),
        [0] * units.kg
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter
        ),
        xr.DataArray(
            np.zeros(5) * units.kg,
            dims=('x',),
            coords={'x': np.linspace(0, 1, 5)}
        )
    ),
    (
        xr.DataArray(
            np.linspace(0, 1, 5) * units.meter,
            attrs={'description': 'A range of values'}
        ),
        np.arange(4, dtype=np.float64)
    )
])
def test_wrap_with_wrap_like_kwarg_raising_dimensionality_error(test, other):
    """Test the preprocess and wrap decorator when a dimensionality error is raised."""
    @preprocess_and_wrap(wrap_like=other, match_unit=True)
    def almost_identity(arg):
        return arg

    with pytest.raises(DimensionalityError):
        almost_identity(test)


def test_wrap_with_argument_kwarg():
    """Test the preprocess and wrap decorator with signature recognition."""
    @preprocess_and_wrap(wrap_like='a')
    def double(a):
        return units.Quantity(2) * a

    test = xr.DataArray([1, 3, 5, 7] * units.m)
    expected = xr.DataArray([2, 6, 10, 14] * units.m)

    xr.testing.assert_identical(double(test), expected)


def test_preprocess_and_wrap_with_broadcasting():
    """Test preprocessing and wrapping decorator with arguments to broadcast specified."""
    # Not quantified
    data = xr.DataArray(np.arange(9).reshape((3, 3)), dims=('y', 'x'), attrs={'units': 'N'})
    # Quantified
    data2 = xr.DataArray([1, 0, 0] * units.m, dims=('y'))

    @preprocess_and_wrap(broadcast=('a', 'b'))
    def func(a, b):
        return a * b

    assert_array_equal(func(data, data2), [[0, 1, 2], [0, 0, 0], [0, 0, 0]] * units('N m'))


def test_preprocess_and_wrap_broadcasting_error():
    """Test that decorator with bad arguments specified to broadcast errors out."""
    with pytest.raises(ValueError):
        @preprocess_and_wrap(broadcast=('a', 'c'))
        def func(a, b):
            """Test a mismatch between arguments in signature and decorator."""


def test_preprocess_and_wrap_with_to_magnitude():
    """Test preprocessing and wrapping with casting to magnitude."""
    data = xr.DataArray([1, 0, 1] * units.m)
    data2 = [0, 1, 1] * units.cm

    @preprocess_and_wrap(wrap_like='a', to_magnitude=True)
    def func(a, b):
        return a * b

    np.testing.assert_array_equal(func(data, data2), np.array([0, 0, 1]))


def test_preprocess_and_wrap_with_variable():
    """Test preprocess and wrapping decorator when using an xarray Variable."""
    data1 = xr.DataArray([1, 0, 1], dims=('x',), attrs={'units': 'meter'})
    data2 = xr.Variable(data=[0, 1, 1], dims=('x',), attrs={'units': 'meter'})

    @preprocess_and_wrap(wrap_like='a')
    def func(a, b):
        return a * b

    # Note, expected units are meter, not meter**2, since attributes are stripped from
    # Variables
    expected_12 = xr.DataArray([0, 0, 1] * units.m, dims=('x',))
    expected_21 = [0, 0, 1] * units.m

    with pytest.warns(UserWarning, match='Argument b given as xarray Variable'):
        result_12 = func(data1, data2)
    with pytest.warns(UserWarning, match='Argument a given as xarray Variable'):
        result_21 = func(data2, data1)

    assert isinstance(result_12, xr.DataArray)
    xr.testing.assert_identical(func(data1, data2), expected_12)
    assert is_quantity(result_21)
    assert_array_equal(func(data2, data1), expected_21)


def test_grid_deltas_from_dataarray_lonlat(test_da_lonlat):
    """Test grid_deltas_from_dataarray with a lonlat grid."""
    dx, dy = grid_deltas_from_dataarray(test_da_lonlat)
    true_dx = np.array([[[321609.59212064, 321609.59212065, 321609.59212064],
                         [310320.85961483, 310320.85961483, 310320.85961483],
                         [297980.72966733, 297980.72966733, 297980.72966733],
                         [284629.6008561, 284629.6008561, 284629.6008561]]]) * units.m
    true_dy = np.array([[[369603.78775948, 369603.78775948, 369603.78775948, 369603.78775948],
                         [369802.28173967, 369802.28173967, 369802.28173967, 369802.28173967],
                         [370009.56291098, 370009.56291098, 370009.56291098,
                          370009.56291098]]]) * units.m
    assert_array_almost_equal(dx, true_dx, 5)
    assert_array_almost_equal(dy, true_dy, 5)


def test_grid_deltas_from_dataarray_xy(test_da_xy):
    """Test grid_deltas_from_dataarray with a xy grid."""
    dx, dy = grid_deltas_from_dataarray(test_da_xy)
    true_dx = np.array([[[[500] * 3]]]) * units('km')
    true_dy = np.array([[[[500]] * 3]]) * units('km')
    assert_array_almost_equal(dx, true_dx, 5)
    assert_array_almost_equal(dy, true_dy, 5)


def test_grid_deltas_from_dataarray_actual_xy(test_da_xy, ccrs):
    """Test grid_deltas_from_dataarray with a xy grid and kind='actual'."""
    # Construct lon/lat coordinates
    y, x = xr.broadcast(*test_da_xy.metpy.coordinates('y', 'x'))
    lon, lat = pyproj.Proj(test_da_xy.metpy.pyproj_crs)(
        x.values,
        y.values,
        inverse=True,
        radians=False
    )
    test_da_xy = test_da_xy.assign_coords(
        longitude=xr.DataArray(lon, dims=('y', 'x'), attrs={'units': 'degrees_east'}),
        latitude=xr.DataArray(lat, dims=('y', 'x'), attrs={'units': 'degrees_north'}))

    # Actually test calculation
    dx, dy = grid_deltas_from_dataarray(test_da_xy, kind='actual')
    true_dx = [[[[494152.626, 493704.152, 492771.132],
                 [498464.118, 498199.037, 497616.042],
                 [499999.328, 499979.418, 499863.087],
                 [498464.608, 498768.783, 499266.193]]]] * units.m
    true_dy = [[[[496587.363, 496410.523, 495857.430, 494863.795],
                 [499498.308, 499429.714, 499191.065, 498689.047],
                 [499474.250, 499549.538, 499727.711, 499874.122]]]] * units.m
    assert_array_almost_equal(dx, true_dx, 2)
    assert_array_almost_equal(dy, true_dy, 2)


def test_grid_deltas_from_dataarray_nominal_lonlat(test_da_lonlat):
    """Test grid_deltas_from_dataarray with a lonlat grid and kind='nominal'."""
    dx, dy = grid_deltas_from_dataarray(test_da_lonlat, kind='nominal')
    true_dx = [[[3.333333] * 3]] * units.degrees
    true_dy = [[[3.333333]] * 3] * units.degrees
    assert_array_almost_equal(dx, true_dx, 5)
    assert_array_almost_equal(dy, true_dy, 5)


def test_grid_deltas_from_dataarray_lonlat_assumed_order():
    """Test grid_deltas_from_dataarray when dim order must be assumed."""
    # Create test dataarray
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

    # Run and check for warning
    with pytest.warns(UserWarning, match=r'y and x dimensions unable to be identified.*'):
        dx, dy = grid_deltas_from_dataarray(test_da)

    # Check results
    true_dx = [[222031.0111961, 222107.8492205],
               [222031.0111961, 222107.8492205],
               [222031.0111961, 222107.8492205]] * units.m
    true_dy = [[175661.5413976, 170784.1311091, 165697.7563223],
               [175661.5413976, 170784.1311091, 165697.7563223]] * units.m
    assert_array_almost_equal(dx, true_dx, 5)
    assert_array_almost_equal(dy, true_dy, 5)


def test_grid_deltas_from_dataarray_invalid_kind(test_da_xy):
    """Test grid_deltas_from_dataarray when kind is invalid."""
    with pytest.raises(ValueError):
        grid_deltas_from_dataarray(test_da_xy, kind='invalid')


def test_add_vertical_dim_from_xarray():
    """Test decorator for automatically determining the vertical dimension number."""
    @add_vertical_dim_from_xarray
    def return_vertical_dim(data, vertical_dim=None):
        return vertical_dim
    test_da = xr.DataArray(np.zeros((2, 2, 2, 2)), dims=('time', 'isobaric', 'y', 'x'))
    assert return_vertical_dim(test_da) == 1
