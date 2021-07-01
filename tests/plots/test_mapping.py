# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Test the handling of various mapping tasks."""

import pytest

ccrs = pytest.importorskip('cartopy.crs')

from metpy.plots.mapping import CFProjection  # noqa: E402


def test_inverse_flattening_0():
    """Test new code for dealing the case where inverse_flattening = 0."""
    attrs = {'grid_mapping_name': 'lambert_conformal_conic', 'semi_major_axis': 6367000,
             'semi_minor_axis': 6367000, 'inverse_flattening': 0}
    proj = CFProjection(attrs)

    crs = proj.to_cartopy()
    globe_params = crs.globe.to_proj4_params()

    assert globe_params['ellps'] == 'sphere'
    assert globe_params['a'] == 6367000
    assert globe_params['b'] == 6367000


def test_cfprojection_arg_mapping():
    """Test the projection mapping arguments."""
    source = {'source': 'a', 'longitude_of_projection_origin': -100}

    # 'dest' should be the argument in the output, with the value from source
    mapping = [('dest', 'source')]

    kwargs = CFProjection.build_projection_kwargs(source, mapping)
    assert kwargs == {'dest': 'a', 'central_longitude': -100}


def test_cfprojection_api():
    """Test the basic API of the projection interface."""
    attrs = {'grid_mapping_name': 'lambert_conformal_conic', 'earth_radius': 6367000}
    proj = CFProjection(attrs)

    assert proj['earth_radius'] == 6367000
    assert proj.to_dict() == attrs
    assert str(proj) == 'Projection: lambert_conformal_conic'


def test_bad_projection_raises():
    """Test behavior when given an unknown projection."""
    attrs = {'grid_mapping_name': 'unknown'}
    with pytest.raises(ValueError) as exc:
        CFProjection(attrs).to_cartopy()

    assert 'Unhandled projection' in str(exc.value)


def test_globe():
    """Test handling building a cartopy globe."""
    attrs = {'grid_mapping_name': 'lambert_conformal_conic', 'earth_radius': 6367000,
             'standard_parallel': 25}
    proj = CFProjection(attrs)

    crs = proj.to_cartopy()
    globe_params = crs.globe.to_proj4_params()

    assert globe_params['ellps'] == 'sphere'
    assert globe_params['a'] == 6367000
    assert globe_params['b'] == 6367000


def test_globe_spheroid():
    """Test handling building a cartopy globe that is not spherical."""
    attrs = {'grid_mapping_name': 'lambert_conformal_conic', 'semi_major_axis': 6367000,
             'semi_minor_axis': 6360000}
    proj = CFProjection(attrs)

    crs = proj.to_cartopy()
    globe_params = crs.globe.to_proj4_params()

    assert 'ellps' not in globe_params
    assert globe_params['a'] == 6367000
    assert globe_params['b'] == 6360000


def test_aea():
    """Test handling albers equal area projection."""
    attrs = {'grid_mapping_name': 'albers_conical_equal_area', 'earth_radius': 6367000,
             'standard_parallel': [20, 50]}
    proj = CFProjection(attrs)

    crs = proj.to_cartopy()
    assert isinstance(crs, ccrs.AlbersEqualArea)
    assert crs.proj4_params['lat_1'] == 20
    assert crs.proj4_params['lat_2'] == 50
    assert crs.globe.to_proj4_params()['ellps'] == 'sphere'


def test_aea_minimal():
    """Test handling albers equal area projection with minimal attributes."""
    attrs = {'grid_mapping_name': 'albers_conical_equal_area'}
    crs = CFProjection(attrs).to_cartopy()
    assert isinstance(crs, ccrs.AlbersEqualArea)


def test_aea_single_std_parallel():
    """Test albers equal area with one standard parallel."""
    attrs = {'grid_mapping_name': 'albers_conical_equal_area', 'standard_parallel': 20}
    crs = CFProjection(attrs).to_cartopy()
    assert isinstance(crs, ccrs.AlbersEqualArea)
    assert crs.proj4_params['lat_1'] == 20


def test_lcc():
    """Test handling lambert conformal conic projection."""
    attrs = {'grid_mapping_name': 'lambert_conformal_conic', 'earth_radius': 6367000,
             'standard_parallel': [25, 30]}
    proj = CFProjection(attrs)

    crs = proj.to_cartopy()
    assert isinstance(crs, ccrs.LambertConformal)
    assert crs.proj4_params['lat_1'] == 25
    assert crs.proj4_params['lat_2'] == 30
    assert crs.globe.to_proj4_params()['ellps'] == 'sphere'


def test_lcc_minimal():
    """Test handling lambert conformal conic projection with minimal attributes."""
    attrs = {'grid_mapping_name': 'lambert_conformal_conic'}
    crs = CFProjection(attrs).to_cartopy()
    assert isinstance(crs, ccrs.LambertConformal)


def test_lcc_single_std_parallel():
    """Test lambert conformal projection with one standard parallel."""
    attrs = {'grid_mapping_name': 'lambert_conformal_conic', 'standard_parallel': 25}
    crs = CFProjection(attrs).to_cartopy()
    assert isinstance(crs, ccrs.LambertConformal)
    assert crs.proj4_params['lat_1'] == 25


def test_mercator():
    """Test handling a mercator projection."""
    attrs = {'grid_mapping_name': 'mercator', 'standard_parallel': 25,
             'longitude_of_projection_origin': -100, 'false_easting': 0, 'false_westing': 0,
             'central_latitude': 0}
    crs = CFProjection(attrs).to_cartopy()

    assert isinstance(crs, ccrs.Mercator)
    assert crs.proj4_params['lat_ts'] == 25
    assert crs.proj4_params['lon_0'] == -100


def test_mercator_scale_factor():
    """Test handling a mercator projection with a scale factor."""
    attrs = {'grid_mapping_name': 'mercator', 'scale_factor_at_projection_origin': 0.9}
    crs = CFProjection(attrs).to_cartopy()

    assert isinstance(crs, ccrs.Mercator)
    assert crs.proj4_params['k_0'] == 0.9


def test_geostationary():
    """Test handling a geostationary projection."""
    attrs = {'grid_mapping_name': 'geostationary', 'perspective_point_height': 35000000,
             'longitude_of_projection_origin': -100, 'sweep_angle_axis': 'x',
             'latitude_of_projection_origin': 0}
    crs = CFProjection(attrs).to_cartopy()

    assert isinstance(crs, ccrs.Geostationary)
    assert crs.proj4_params['h'] == 35000000
    assert crs.proj4_params['lon_0'] == -100
    assert crs.proj4_params['sweep'] == 'x'


def test_geostationary_fixed_angle():
    """Test handling geostationary information that gives fixed angle instead of sweep."""
    attrs = {'grid_mapping_name': 'geostationary', 'fixed_angle_axis': 'y'}
    crs = CFProjection(attrs).to_cartopy()

    assert isinstance(crs, ccrs.Geostationary)
    assert crs.proj4_params['sweep'] == 'x'


def test_stereographic():
    """Test handling a stereographic projection."""
    attrs = {'grid_mapping_name': 'stereographic', 'scale_factor_at_projection_origin': 0.9,
             'longitude_of_projection_origin': -100, 'latitude_of_projection_origin': 60}
    crs = CFProjection(attrs).to_cartopy()

    assert isinstance(crs, ccrs.Stereographic)
    assert crs.proj4_params['lon_0'] == -100
    assert crs.proj4_params['lat_0'] == 60
    assert crs.proj4_params['k_0'] == 0.9


def test_polar_stereographic():
    """Test handling a polar stereographic projection."""
    attrs = {'grid_mapping_name': 'polar_stereographic', 'latitude_of_projection_origin': 90,
             'scale_factor_at_projection_origin': 0.9,
             'straight_vertical_longitude_from_pole': -100, }
    crs = CFProjection(attrs).to_cartopy()

    assert isinstance(crs, ccrs.Stereographic)
    assert crs.proj4_params['lon_0'] == -100
    assert crs.proj4_params['lat_0'] == 90
    assert crs.proj4_params['k_0'] == 0.9


def test_polar_stereographic_std_parallel():
    """Test handling a polar stereographic projection that gives a standard parallel."""
    attrs = {'grid_mapping_name': 'polar_stereographic', 'latitude_of_projection_origin': -90,
             'standard_parallel': 60}
    crs = CFProjection(attrs).to_cartopy()

    assert isinstance(crs, ccrs.Stereographic)
    assert crs.proj4_params['lat_0'] == -90
    assert crs.proj4_params['lat_ts'] == 60


def test_lat_lon():
    """Test handling basic lat/lon projection."""
    attrs = {'grid_mapping_name': 'latitude_longitude'}
    crs = CFProjection(attrs).to_cartopy()
    assert isinstance(crs, ccrs.PlateCarree)


def test_eq():
    """Test that two CFProjection instances are equal given that they have the same attrs."""
    attrs = {'grid_mapping_name': 'latitude_longitude'}
    cf_proj_1 = CFProjection(attrs)
    cf_proj_2 = CFProjection(attrs.copy())
    assert cf_proj_1 == cf_proj_2


def test_ne():
    """Test that two CFProjection instances are not equal when attrs differs."""
    cf_proj_1 = CFProjection({'grid_mapping_name': 'latitude_longitude'})
    cf_proj_2 = CFProjection({'grid_mapping_name': 'lambert_conformal_conic'})
    assert cf_proj_1 != cf_proj_2
