# Copyright (c) 2008,2015,2016,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from test_thermo for xarray parcel functions.

import metpy
import numpy as np
import pytest
import xarray as xarray
import modules.parcel_functions as parcel
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from metpy.units import units

# Load xarray-parcel's lookup tables.
parcel.load_moist_adiabat_lookups()

def run_all_tests():
    """Run all the tests."""
    test_dry_lapse()
    test_dry_lapse_2_levels()
    test_moist_lapse()
    test_moist_lapse_ref_pres()
    test_moist_lapse_scalar()
    test_moist_lapse_uniform()
    test_parcel_profile()
    test_parcel_profile_lcl()
    test_parcel_profile_saturated()
    test_lcl()
    # test_lcl_nans() -- metpy lcl implementation sometimes fails to converge.
    test_lfc_basic()
    test_lfc_ml()
    test_lfc_ml2()
    test_lfc_intersection()
    test_no_lfc()
    test_lfc_inversion()
    test_lfc_equals_lcl()
    test_sensitive_sounding()
    test_lfc_sfc_precision()
    test_lfc_pos_area_below_lcl()
    test_el()
    test_el_ml()
    test_no_el()
    test_no_el_multi_crossing()
    test_lfc_and_el_below_lcl()
    test_el_lfc_equals_lcl()
    test_el_small_surface_instability()
    test_no_el_parcel_colder()
    test_el_below_lcl()
    test_cape_cin()
    test_cape_cin_no_el()
    test_cape_cin_no_lfc()
    test_most_unstable_parcel()
    test_surface_based_cape_cin()
    test_profile_with_nans()
    test_most_unstable_cape_cin_surface()
    test_mixed_parcel()
    test_mixed_layer_cape_cin()
    test_multiple_lfcs_el_simple()
    test_mixed_layer()
    test_lfc_not_below_lcl()
    test_cape_cin_custom_profile()
    test_parcel_profile_below_lcl()
    test_lcl_convergence_issue()
    test_cape_cin_value_error()
    test_lcl_grid_surface_lcls()
    test_lifted_index()
    print('All tests passed.')
    
def run_moist_lapse_tests_looser():
    """Run all the tests, with looser matching requirements."""
    test_moist_lapse()
    test_moist_lapse_ref_pres()
    test_moist_lapse_scalar()
    test_moist_lapse_uniform(dp=2)
    print('Moist lapse tests passed.')

def metpy_moist_lapse(pressure, parcel_temperature, parcel_pressure=None):
    """A wrapper for metpy's moist_lapse()."""
    
    if isinstance(parcel_pressure, xarray.DataArray):
        parcel_pressure = parcel_pressure.values
    if isinstance(pressure, xarray.DataArray):
        pressure = pressure.values
    if isinstance(parcel_temperature, xarray.DataArray):
        parcel_temperature = parcel_temperature.values
        
    if not parcel_pressure is None:
        parcel_pressure = parcel_pressure * units.hPa
        
    return metpy.calc.moist_lapse(pressure=pressure*units.hPa, 
                                  temperature=parcel_temperature*units.K, 
                                  reference_pressure=parcel_pressure).m

def vert_array(x, units):
    """Make an xarray object with one dimension."""
    return xarray.DataArray(x, dims='model_level_number', 
                            coords={'model_level_number': np.arange(1,len(x)+1)},
                            attrs={'units': units})

def test_dry_lapse():
    """Test dry_lapse calculation."""
    levels = vert_array([1000, 900, 864.89], 'hPa')
    temps = parcel.dry_lapse(pressure=levels, parcel_temperature=303.15)
    assert_array_almost_equal(temps.values, np.array([303.15, 294.16, 290.83]), decimal=2)

def test_dry_lapse_2_levels():
    """Test dry_lapse calculation when given only two levels."""
    levels = vert_array([1000., 500.], 'hPa')
    temps = parcel.dry_lapse(pressure=levels, parcel_temperature=293.)
    assert_array_almost_equal(temps, [293., 240.3583], 4)

def test_moist_lapse():
    """Test moist_lapse calculation."""
    levels = vert_array([1000., 800., 600., 500., 400.], 'hPa')
    temp = parcel.moist_lapse(pressure=levels, parcel_temperature=293.)
    assert_array_almost_equal(temp, [293, 284.64, 272.81, 264.42, 252.91], 2)

def test_moist_lapse_ref_pres():
    """Test moist_lapse with a reference pressure."""
    levels = vert_array([1050., 800., 600., 500., 400.], 'hPa')
    temp = parcel.moist_lapse(pressure=levels, parcel_temperature=293, parcel_pressure=1000)
    assert_array_almost_equal(temp, [294.76, 284.64, 272.81, 264.42, 252.91], 2)

def test_moist_lapse_scalar():
    """Test moist_lapse when given a scalar desired pressure and a reference pressure."""
    levels = vert_array([800.], 'hPa')
    temp = parcel.moist_lapse(pressure=levels, parcel_temperature=293, parcel_pressure=1000)
    assert_array_almost_equal(temp, [284.64], 2)

def test_moist_lapse_uniform(dp=7):
    """Test moist_lapse when given a uniform array of pressures."""
    levels = vert_array([900., 900., 900.], 'hPa')
    temp = parcel.moist_lapse(pressure=levels, parcel_temperature=293.15)
    assert_almost_equal(temp, np.array([293.15, 293.15, 293.15]), dp)

def test_parcel_profile(dp=2):
    """Test parcel profile calculation."""
    levels = vert_array([1000., 900., 800., 700., 600., 500., 400.], 'hPa')
    true_prof = np.array([303.15, 294.16, 288.026, 283.073, 277.058, 269.402, 258.966])

    parcel_temperature = xarray.DataArray(303.15, attrs={'units': 'K'})
    parcel_pressure = xarray.DataArray(1000, attrs={'units': 'hPa'})
    parcel_dewpoint = xarray.DataArray(293.15, attrs={'units': 'K'})
    
    prof = parcel.parcel_profile(pressure=levels, parcel_pressure=parcel_pressure,
                                 parcel_temperature=parcel_temperature, 
                                 parcel_dewpoint=parcel_dewpoint)
    
    assert_array_almost_equal(prof.temperature, true_prof, dp)

def test_parcel_profile_lcl(dp=3):
    """Test parcel profile with lcl calculation."""
    p = vert_array([1004., 1000., 943., 928., 925., 850., 839., 749., 700., 699.], 'hPa')
    t = vert_array([24.2, 24., 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2, 13.], 'K') + 273.15
    
    true_t = np.array([24.2, 24., 22.047, 20.2, 21.6, 21.4, 20.4, 20.2, 14.4, 13.2, 13.]) + 273.15
    true_p = np.array([1004., 1000., 970.711, 943., 928., 925., 850., 839., 749., 700., 699.])
    true_prof = np.array([297.35, 297.01, 294.5, 293.48, 292.92, 292.81, 289.79, 289.32,
                          285.15, 282.59, 282.53])

    parcel_temperature = xarray.DataArray(24.2+273.15, attrs={'units': 'K'})
    parcel_pressure = xarray.DataArray(1004, attrs={'units': 'hPa'})
    parcel_dewpoint = xarray.DataArray(21.9+273.15, attrs={'units': 'K'})
    
    prof = parcel.parcel_profile(pressure=p, 
                                 parcel_pressure=parcel_pressure, 
                                 parcel_temperature=parcel_temperature,
                                 parcel_dewpoint=parcel_dewpoint)
    prof = parcel.add_lcl_to_profile(profile=prof, temperature=t)
    
    assert_array_almost_equal(prof.pressure, true_p, dp)
    assert_array_almost_equal(prof.environment_temperature, true_t, dp)
    assert_array_almost_equal(prof.temperature, true_prof, dp-1)

def test_parcel_profile_saturated(dp=2):
    """Test parcel_profile works when LCL in levels (issue #232)."""
    levels = vert_array([1000., 700., 500.], 'hPa')
    true_prof = np.array([296.95, 284.381, 271.123])

    parcel_temperature = xarray.DataArray(23.8+273.15, attrs={'units': 'K'})
    parcel_pressure = xarray.DataArray(1000, attrs={'units': 'hPa'})
    parcel_dewpoint = xarray.DataArray(23.8+273.15, attrs={'units': 'K'})
    
    prof = parcel.parcel_profile(pressure=levels, 
                                 parcel_pressure=parcel_pressure, 
                                 parcel_temperature=parcel_temperature,
                                 parcel_dewpoint=parcel_dewpoint)
    assert_array_almost_equal(prof.temperature, true_prof, dp)

def test_lcl():
    """Test LCL calculation."""
    parcel_temperature = xarray.DataArray(30+273.15, attrs={'units': 'K'})
    parcel_pressure = xarray.DataArray(1000, attrs={'units': 'hPa'})
    parcel_dewpoint = xarray.DataArray(20+273.15, attrs={'units': 'K'})
    lcl = parcel.lcl(parcel_pressure=parcel_pressure, 
                     parcel_temperature=parcel_temperature, 
                     parcel_dewpoint=parcel_dewpoint)
    assert_almost_equal(lcl.lcl_pressure, 864.806, 2)
    assert_almost_equal(lcl.lcl_temperature, 17.676+273.15, 2)

def test_lcl_nans():
    """Test LCL calculation on data with nans."""
    p = xarray.DataArray([900., 900., 900., 900.], attrs={'units': 'hPa'}) 
    t = xarray.DataArray(np.array([np.nan, 25., 25., 25.])+273.15, attrs={'units': 'K'})
    d = xarray.DataArray(np.array([20., 20., np.nan, 20.])+273.15, attrs={'units': 'K'})
    lcl = parcel.lcl(parcel_pressure=p, parcel_temperature=t, parcel_dewpoint=d)

    assert_array_almost_equal(lcl.lcl_pressure, 
                              np.array([np.nan, 836.4098648012595, 
                                        np.nan, 836.4098648012595]))
    assert_array_almost_equal(lcl.lcl_temperature, 
                              np.array([np.nan, 18.82281982535794,
                                        np.nan, 18.82281982535794])+273.15)

def test_lfc_basic(dp=2):
    """Test LFC calculation."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -49.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc = parcel.lfc_el(profile=profile)
    assert_almost_equal(lfc.lfc_pressure, 727.371, dp)
    assert_almost_equal(lfc.lfc_temperature, 9.705+273.15, dp)

def test_lfc_ml(dp=2):
    """Test Mixed-Layer LFC calculation."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    levels.name = 'pressure'
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -49.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
    mixed = parcel.mixed_parcel(pressure=levels, temperature=temperatures, dewpoint=dewpoints)
    mixed_parcel_prof = parcel.parcel_profile_with_lcl(pressure=levels, 
                                                       temperature=temperatures,
                                                       parcel_pressure=mixed.pressure,
                                                       parcel_temperature=mixed.temperature,
                                                       parcel_dewpoint=mixed.dewpoint)
    lfc = parcel.lfc_el(profile=mixed_parcel_prof)
    assert_almost_equal(lfc.lfc_pressure, 601.225, dp)
    assert_almost_equal(lfc.lfc_temperature, -1.90688+273.15, dp)

def test_lfc_ml2():
    """Test a mixed-layer LFC calculation that previously crashed."""
    levels = vert_array([1024.95703125, 1016.61474609, 1005.33056641, 991.08544922, 973.4163208,
                         951.3381958, 924.82836914, 898.25482178, 873.46124268, 848.69830322,
                         823.92553711, 788.49304199, 743.44580078, 700.50970459, 659.62017822,
                         620.70861816, 583.69421387, 548.49719238, 515.03826904, 483.24401855,
                         453.0418396, 424.36477661, 397.1505127, 371.33441162, 346.85922241,
                         323.66995239, 301.70935059, 280.92651367, 261.27053833, 242.69168091,
                         225.14237976, 208.57781982, 192.95333862, 178.22599792, 164.39630127,
                         151.54336548, 139.68635559, 128.74923706, 118.6588974, 109.35111237,
                         100.76405334, 92.84288025, 85.53556824, 78.79430389, 72.57549286,
                         66.83885193, 61.54678726, 56.66480637, 52.16108322], 'hPa')
    levels.name = 'pressure'
    temperatures = vert_array(np.array([6.00750732, 5.14892578, 4.177948, 3.00268555, 1.55535889,
                                        -0.25527954, -1.93988037, -3.57766724, -4.40600586, -4.19238281,
                                        -3.71185303, -4.47943115, -6.81280518, -8.08685303, -8.41287231,
                                        -10.79302979, -14.13262939, -16.85784912, -19.51675415,
                                        -22.28689575, -24.99938965, -27.79664612, -30.90414429,
                                        -34.49435425, -38.438797, -42.27981567, -45.99230957,
                                        -49.75340271, -53.58230591, -57.30686951, -60.76026917,
                                        -63.92070007, -66.72470093, -68.97846985, -70.4264679,
                                        -71.16407776, -71.53797913, -71.64375305, -71.52735901,
                                        -71.53523254, -71.61097717, -71.92687988, -72.68682861,
                                        -74.129776, -76.02471924, -76.88977051, -76.26008606,
                                        -75.90351868, -76.15809631])+273.15, 'K')
    dewpoints = vert_array(np.array([4.50012302, 3.42483997, 2.78102994, 2.24474645, 1.593485, -0.9440815,
                                     -3.8044982, -3.55629468, -9.7376976, -10.2950449, -9.67498302,
                                     -10.30486488, -8.70559597, -8.71669006, -12.66509628, -18.6697197,
                                     -23.00351334, -29.46240425, -36.82178497, -41.68824768, -44.50320816,
                                     -48.54426575, -52.50753403, -51.09564209, -48.92690659, -49.97380829,
                                     -51.57516098, -52.62096405, -54.24332809, -57.09109879, -60.5596199,
                                     -63.93486404, -67.07530212, -70.01263428, -72.9258728, -76.12271881,
                                     -79.49847412, -82.2350769, -83.91127014, -84.95665741, -85.61238861,
                                     -86.16391754, -86.7653656, -87.34436035, -87.87495422, -88.34281921,
                                     -88.74453735, -89.04680634, -89.26436615])+273.15, 'K')
    mixed = parcel.mixed_parcel(pressure=levels, temperature=temperatures, dewpoint=dewpoints)
    mixed_parcel_prof = parcel.parcel_profile_with_lcl(pressure=levels, 
                                                       temperature=temperatures,
                                                       parcel_pressure=mixed.pressure,
                                                       parcel_temperature=mixed.temperature,
                                                       parcel_dewpoint=mixed.dewpoint)
    lfc = parcel.lfc_el(profile=mixed_parcel_prof)
    assert_almost_equal(lfc.lfc_pressure, 962.34, 2)
    assert_almost_equal(lfc.lfc_temperature, 0.767+273.15, 2)

def test_lfc_intersection(dp=2):
    """Test LFC calculation when LFC is below a tricky intersection."""
    levels = vert_array([1024.957, 930., 924.828, 898.255, 873.461, 848.698, 823.926, 788.493], 'hPa')
    levels.name = 'pressure'
    temperatures = vert_array(np.array([6.008, -10., -6.94, -8.58, -4.41, -4.19, -3.71, -4.48])+273.15, 'K')
    dewpoints = vert_array(np.array([5., -10., -7., -9., -4.5, -4.2, -3.8, -4.5])+273.15, 'K')
    
    mixed = parcel.mixed_parcel(pressure=levels, temperature=temperatures, dewpoint=dewpoints)
    
    # Calculate parcel profile without LCL, as per metpy unit tests.
    mixed_parcel_prof = parcel.parcel_profile(pressure=levels, 
                                              parcel_pressure=mixed.pressure,
                                              parcel_temperature=mixed.temperature,
                                              parcel_dewpoint=mixed.dewpoint)
    mixed_parcel_prof['environment_temperature'] = temperatures
    lfc = parcel.lfc_el(profile=mixed_parcel_prof)
    assert_almost_equal(lfc.lfc_pressure, 981.620, dp)

def test_no_lfc():
    """Test LFC calculation when there is no LFC in the data."""
    levels = vert_array([959., 867.9, 779.2, 647.5, 472.5, 321.9, 251.], 'hPa')
    temperatures = vert_array(np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5])+273.15, 'K')
    dewpoints = vert_array(np.array([9., 4.3, -21.2, -26.7, -31., -53.3, -66.7])+273.15, 'K')
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc = parcel.lfc_el(profile=profile)
    assert(np.isnan(lfc.lfc_pressure))
    assert(np.isnan(lfc.lfc_temperature))

def test_lfc_inversion(dp=2):
    """Test LFC when there is an inversion to be sure we don't pick that."""
    levels = vert_array([963., 789., 782.3, 754.8, 728.1, 727., 700.,
                         571., 450., 300., 248.], 'hPa')
    temperatures = vert_array(np.array([25.4, 18.4, 17.8, 15.4, 12.9, 12.8,
                                        10., -3.9, -16.3, -41.1, -51.5])+273.15, 'K')
    dewpoints = vert_array(np.array([20.4, 0.4, -0.5, -4.3, -8., -8.2, -9.,
                                     -23.9, -33.3, -54.1, -63.5])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc = parcel.lfc_el(profile=profile)
    
    assert_almost_equal(lfc.lfc_pressure, 705.8806 , dp)
    assert_almost_equal(lfc.lfc_temperature, 10.6232+273.15, dp)

def test_lfc_equals_lcl():
    """Test LFC when there is no cap and the lfc is equal to the lcl."""
    levels = vert_array([912., 905.3, 874.4, 850., 815.1, 786.6, 759.1,
                         748., 732.2, 700., 654.8], 'hPa')
    temperatures = vert_array(np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8,
                                        14.0, 13.2, 12.6, 11.4, 7.1])+273.15, 'K')
    dewpoints = vert_array(np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6,
                                     8.8, 0., -18.6, -22.9])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc = parcel.lfc_el(profile=profile)
    assert_almost_equal(lfc.lfc_pressure, 777.0786, 2)
    assert_almost_equal(lfc.lfc_temperature, 15.8714+273.15, 2)

def test_sensitive_sounding(dp=2):
    """Test quantities for a sensitive sounding (#902)."""
    # This sounding has a very small positive area in the low level. It's only captured
    # properly if the parcel profile includes the LCL, otherwise it breaks LFC and CAPE
    levels = vert_array([1004., 1000., 943., 928., 925., 850., 839., 749., 700., 699.,
                         603., 500., 404., 400., 363., 306., 300., 250., 213., 200.,
                         176., 150.], 'hPa')
    temperatures = vert_array(np.array([24.2, 24., 20.2, 21.6, 21.4, 20.4, 20.2, 14.4,
                                        13.2, 13., 6.8, -3.3, -13.1, -13.7, -17.9, -25.5, 
                                        -26.9, -37.9, -46.7, -48.7, -52.1, -58.9])+273.15, 'K')
    dewpoints = vert_array(np.array([21.9, 22.1, 19.2, 20.5, 20.4, 18.4, 17.4, 8.4, -2.8, 
                                     -3.0, -15.2, -20.3, -29.1, -27.7, -24.9, -39.5, -41.9,
                                     -51.9, -60.7, -62.7, -65.1, -71.9])+273.15, 'K')
   
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc = parcel.lfc_el(profile=profile)
    
    assert_almost_equal(lfc.lfc_pressure, 947.422, dp)
    assert_almost_equal(lfc.lfc_temperature, 20.498+273.15, dp)

    cape_cin = parcel.surface_based_cape_cin(pressure=levels, temperature=temperatures, 
                                             dewpoint=dewpoints)
    assert_almost_equal(cape_cin.cape, 0.1115, 3)
    assert_almost_equal(cape_cin.cin, -6.0866, 3)

def test_lfc_sfc_precision():
    """Test LFC when there are precision issues with the parcel path."""
    levels = vert_array([839., 819.4, 816., 807., 790.7, 763., 736.2, 722., 710.1, 700.], 'hPa')
    temperatures = vert_array(np.array([20.6, 22.3, 22.6, 22.2, 20.9, 18.7, 16.4, 
                                        15.2, 13.9, 12.8])+273.15, 'K')
    dewpoints = vert_array(np.array([10.6, 8., 7.6, 6.2, 5.7, 4.7, 3.7, 3.2, 3., 2.8])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc = parcel.lfc_el(profile=profile)
    assert(np.isnan(lfc.lfc_pressure))
    assert(np.isnan(lfc.lfc_temperature))

def test_lfc_pos_area_below_lcl():
    """Test LFC when there is positive area below the LCL (#1003)."""
    levels = vert_array([902.1554, 897.9034, 893.6506, 889.4047, 883.063, 874.6284, 866.2387, 857.887,
                         849.5506, 841.2686, 833.0042, 824.7891, 812.5049, 796.2104, 776.0027, 751.9025,
                         727.9612, 704.1409, 680.4028, 656.7156, 629.077, 597.4286, 565.6315, 533.5961,
                         501.2452, 468.493, 435.2486, 401.4239, 366.9387, 331.7026, 295.6319, 258.6428,
                         220.9178, 182.9384, 144.959, 106.9778, 69.00213], 'hPa')
    temperatures = vert_array(np.array([-3.039381, -3.703779, -4.15996, -4.562574, -5.131827, -5.856229, -6.568434,
                                        -7.276881, -7.985013, -8.670911, -8.958063, -7.631381, -6.05927, -5.083627,
                                        -5.11576, -5.687552, -5.453021, -4.981445, -5.236665, -6.324916, -8.434324,
                                        -11.58795, -14.99297, -18.45947, -21.92021, -25.40522, -28.914, -32.78637,
                                        -37.7179, -43.56836, -49.61077, -54.24449, -56.16666, -57.03775, -58.28041,
                                        -60.86264, -64.21677])+273.15, 'K')
    dewpoints = vert_array(np.array([-22.08774, -22.18181, -22.2508, -22.31323, -22.4024, -22.51582, -22.62526,
                                     -22.72919, -22.82095, -22.86173, -22.49489, -21.66936, -21.67332, -21.94054,
                                     -23.63561, -27.17466, -31.87395, -38.31725, -44.54717, -46.99218, -43.17544,
                                     -37.40019, -34.3351, -36.42896, -42.1396, -46.95909, -49.36232, -48.94634,
                                     -47.90178, -49.97902, -55.02753, -63.06276, -72.53742, -88.81377, -93.54573,
                                     -92.92464, -91.57479])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc = parcel.lfc_el(profile=profile)
    assert(np.isnan(lfc.lfc_pressure))
    assert(np.isnan(lfc.lfc_temperature))

def test_el():
    """Test equilibrium layer calculation."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -38.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    assert_almost_equal(el.el_pressure, 471.83286, 3)
    assert_almost_equal(el.el_temperature, -11.5603+273.15, 3)

def test_el_ml():
    """Test equilibrium layer calculation for a mixed parcel."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 400., 269.], 'hPa')
    levels.name = 'pressure'
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -25., -35.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -35., -53.2])+273.15, 'K')
    
    mixed = parcel.mixed_parcel(pressure=levels, temperature=temperatures, dewpoint=dewpoints)
    mixed_parcel_prof = parcel.parcel_profile_with_lcl(pressure=levels, 
                                                       temperature=temperatures,
                                                       parcel_pressure=mixed.pressure,
                                                       parcel_temperature=mixed.temperature,
                                                       parcel_dewpoint=mixed.dewpoint)
    el = parcel.lfc_el(profile=mixed_parcel_prof)
    assert_almost_equal(el.el_pressure, 350.0561, 3)
    assert_almost_equal(el.el_temperature, -28.36156+273.15, 3)

def test_no_el():
    """Test equilibrium layer calculation when there is no EL in the data."""
    levels = vert_array([959., 867.9, 779.2, 647.5, 472.5, 321.9, 251.], 'hPa')
    temperatures = vert_array(np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5])+273.15, 'K')
    dewpoints = vert_array(np.array([19., 14.3, -11.2, -16.7, -21., -43.3, -56.7])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    assert(np.isnan(el.el_pressure))
    assert(np.isnan(el.el_temperature))

def test_no_el_multi_crossing():
    """Test el calculation with no el and severel parcel path-profile crossings."""
    levels = vert_array([918., 911., 880., 873.9, 850., 848., 843.5, 818., 813.8, 785.,
                         773., 763., 757.5, 730.5, 700., 679., 654.4, 645.,
                         643.9], 'hPa')
    temperatures = vert_array(np.array([24.2, 22.8, 19.6, 19.1, 17., 16.8, 16.5, 15., 14.9, 14.4, 16.4,
                                        16.2, 15.7, 13.4, 10.6, 8.4, 5.7, 4.6, 4.5])+273.15, 'K')
    dewpoints = vert_array(np.array([19.5, 17.8, 16.7, 16.5, 15.8, 15.7, 15.3, 13.1, 12.9, 11.9, 6.4,
                                     3.2, 2.6, -0.6, -4.4, -6.6, -9.3, -10.4, -10.5])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    assert(np.isnan(el.el_pressure))
    assert(np.isnan(el.el_temperature))
    
def test_lfc_and_el_below_lcl():
    """Test that LFC and EL are returned as NaN if both are below LCL."""
    dewpoints = vert_array([264.5351, 261.13443, 259.0122, 252.30063, 248.58017, 242.66582], 'K')
    temperatures = vert_array([273.09723, 268.40173, 263.56207, 260.257, 256.63538, 252.91345], 'K')
    levels = vert_array([1017.16, 950, 900, 850, 800, 750], 'hPa')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    assert(np.isnan(el.el_pressure))
    assert(np.isnan(el.el_temperature))
    assert(np.isnan(el.lfc_pressure))
    assert(np.isnan(el.lfc_temperature))
    
def test_el_lfc_equals_lcl():
    """Test equilibrium layer calculation when the lfc equals the lcl."""
    levels = vert_array([912., 905.3, 874.4, 850., 815.1, 786.6, 759.1, 748.,
                         732.3, 700., 654.8, 606.8, 562.4, 501.8, 500., 482.,
                         400., 393.3, 317.1, 307., 300., 252.7, 250., 200.,
                         199.3, 197., 190., 172., 156.6, 150., 122.9, 112.,
                         106.2, 100.], 'hPa')
    temperatures = vert_array(np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8, 14.3,
                                        13.2, 12.6, 11.4, 7.1, 2.2, -2.7, -10.1,
                                        -10.3, -12.4, -23.3, -24.4, -38., -40.1, -41.1,
                                        -49.8, -50.3, -59.1, -59.1, -59.3, -59.7, -56.3,
                                        -56.9, -57.1, -59.1, -60.1, -58.6, -56.9])+273.15, 'K')
    dewpoints = vert_array(np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6, 8.8, 0.,
                                     -18.6, -22.9, -27.8, -32.7, -40.1, -40.3, -42.4, -53.3,
                                     -54.4, -68., -70.1, -70., -70., -70., -70., -70., -70.,
                                     -70., -70., -70., -70., -70., -70., -70., -70.])+273.15, 'K')
    
        
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    
    assert_almost_equal(el.el_pressure, 175.7663, 3)
    assert_almost_equal(el.el_temperature, -57.03994+273.15, 3)

def test_el_small_surface_instability():
    """Test that no EL is found when there is a small pocket of instability at the sfc."""
    levels = vert_array([959., 931.3, 925., 899.3, 892., 867.9, 850., 814.,
                         807.9, 790., 779.2, 751.3, 724.3, 700., 655., 647.5,
                         599.4, 554.7, 550., 500.], 'hPa')
    temperatures = vert_array(np.array([22.2, 20.2, 19.8, 18.4, 18., 17.4, 17., 15.4, 15.4,
                                        15.6, 14.6, 12., 9.4, 7., 2.2, 1.4, -4.2, -9.7,
                                        -10.3, -14.9])+273.15, 'K')
    dewpoints = vert_array(np.array([20., 18.5, 18.1, 17.9, 17.8, 15.3, 13.5, 6.4, 2.2,
                                     -10.4, -10.2, -9.8, -9.4, -9., -15.8, -15.7, -14.8, -14.,
                                     -13.9, -17.9])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    assert(np.isnan(el.el_pressure))
    assert(np.isnan(el.el_temperature))

def test_no_el_parcel_colder():
    """Test no EL when parcel stays colder than environment. INL 20170925-12Z."""
    levels = vert_array([974., 946., 925., 877.2, 866., 850., 814.6, 785.,
                         756.6, 739., 729.1, 700., 686., 671., 641., 613.,
                         603., 586., 571., 559.3, 539., 533., 500., 491.,
                         477.9, 413., 390., 378., 345., 336.], 'hPa')
    temperatures = vert_array(np.array([10., 8.4, 7.6, 5.9, 7.2, 7.6, 6.8, 7.1, 7.7,
                                        7.8, 7.7, 5.6, 4.6, 3.4, 0.6, -0.9, -1.1, -3.1,
                                        -4.7, -4.7, -6.9, -7.5, -11.1, -10.9, -12.1, -20.5, -23.5,
                                        -24.7, -30.5, -31.7])+273.15, 'K')
    dewpoints = vert_array(np.array([8.9, 8.4, 7.6, 5.9, 7.2, 7., 5., 3.6, 0.3,
                                     -4.2, -12.8, -12.4, -8.4, -8.6, -6.4, -7.9, -11.1, -14.1,
                                     -8.8, -28.1, -18.9, -14.5, -15.2, -15.1, -21.6, -41.5, -45.5,
                                     -29.6, -30.6, -32.1])+273.15, 'K')
                        
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    assert(np.isnan(el.el_pressure))
    assert(np.isnan(el.el_temperature))
    
def test_el_below_lcl():
    """Test LFC when there is positive area below the LCL (#1003)."""
    levels = vert_array([902.1554, 897.9034, 893.6506, 889.4047, 883.063, 874.6284, 866.2387, 857.887,
                         849.5506, 841.2686, 833.0042, 824.7891, 812.5049, 796.2104, 776.0027, 751.9025,
                         727.9612, 704.1409, 680.4028, 656.7156, 629.077, 597.4286, 565.6315, 533.5961,
                         501.2452, 468.493, 435.2486, 401.4239, 366.9387, 331.7026, 295.6319, 258.6428,
                         220.9178, 182.9384, 144.959, 106.9778, 69.00213], 'hPa')
    temperatures = vert_array(np.array([-3.039381, -3.703779, -4.15996, -4.562574, -5.131827, -5.856229, -6.568434,
                                        -7.276881, -7.985013, -8.670911, -8.958063, -7.631381, -6.05927, -5.083627,
                                        -5.11576, -5.687552, -5.453021, -4.981445, -5.236665, -6.324916, -8.434324,
                                        -11.58795, -14.99297, -18.45947, -21.92021, -25.40522, -28.914, -32.78637,
                                        -37.7179, -43.56836, -49.61077, -54.24449, -56.16666, -57.03775, -58.28041,
                                        -60.86264, -64.21677])+273.15, 'K')
    dewpoints = vert_array(np.array([-22.08774, -22.18181, -22.2508, -22.31323, -22.4024, -22.51582, -22.62526,
                                     -22.72919, -22.82095, -22.86173, -22.49489, -21.66936, -21.67332, -21.94054,
                                     -23.63561, -27.17466, -31.87395, -38.31725, -44.54717, -46.99218, -43.17544,
                                     -37.40019, -34.3351, -36.42896, -42.1396, -46.95909, -49.36232, -48.94634,
                                     -47.90178, -49.97902, -55.02753, -63.06276, -72.53742, -88.81377, -93.54573,
                                     -92.92464, -91.57479])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    el = parcel.lfc_el(profile=profile)
    assert(np.isnan(el.el_pressure))
    assert(np.isnan(el.el_temperature))
    
def test_cape_cin():
    """Test the basic CAPE and CIN calculation."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -38.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
    
    # Calculate parcel profile without LCL, as per metpy unit tests.
    profile = parcel.parcel_profile(pressure=levels, 
                                    parcel_pressure=levels[0], 
                                    parcel_temperature=temperatures[0],
                                    parcel_dewpoint=dewpoints[0])
    profile['environment_temperature'] = temperatures
    lfc = parcel.lfc_el(profile=profile)
    cape_cin = parcel.cape_cin_base(pressure=levels, 
                                    temperature=temperatures,
                                    lfc_pressure=lfc.lfc_pressure,
                                    el_pressure=lfc.el_pressure,
                                    parcel_profile=profile)
                                
    assert_almost_equal(cape_cin.cape, 75.05354, 2)
    assert_almost_equal(cape_cin.cin, -89.890078, 2)

def test_cape_cin_no_el():
    """Test that CAPE works with no EL."""
    levels = vert_array([959., 779.2, 751.3, 724.3], 'hPa')
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4])+273.15, 'K')
    
    # Calculate parcel profile without LCL, as per metpy unit tests.
    profile = parcel.parcel_profile(pressure=levels, 
                                    parcel_pressure=levels[0], 
                                    parcel_temperature=temperatures[0],
                                    parcel_dewpoint=dewpoints[0])
    profile['environment_temperature'] = temperatures
    lfc = parcel.lfc_el(profile=profile)
    cape_cin = parcel.cape_cin_base(pressure=levels, 
                                    temperature=temperatures,
                                    lfc_pressure=lfc.lfc_pressure,
                                    el_pressure=lfc.el_pressure,
                                    parcel_profile=profile)
    
    assert_almost_equal(cape_cin.cape, 0.08610409, 2)
    assert_almost_equal(cape_cin.cin, -89.8900784, 2)

def test_cape_cin_no_lfc():
    """Test that CAPE is zero with no LFC."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    temperatures = vert_array(np.array([22.2, 24.6, 22., 20.4, 18., -10.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
        
    # Calculate parcel profile without LCL, as per metpy unit tests.
    profile = parcel.parcel_profile(pressure=levels, 
                                    parcel_pressure=levels[0], 
                                    parcel_temperature=temperatures[0],
                                    parcel_dewpoint=dewpoints[0])
    profile['environment_temperature'] = temperatures
    lfc = parcel.lfc_el(profile=profile)
    cape_cin = parcel.cape_cin_base(pressure=levels, 
                                    temperature=temperatures,
                                    lfc_pressure=lfc.lfc_pressure,
                                    el_pressure=lfc.el_pressure,
                                    parcel_profile=profile)
    
    assert_almost_equal(cape_cin.cape, 0.0, 2)
    assert_almost_equal(cape_cin.cin, 0.0, 2)

def test_most_unstable_parcel():
    """Test calculating the most unstable parcel."""
    levels = vert_array([1000., 959., 867.9], 'hPa')
    levels.name = 'pressure'
    temperatures = vert_array(np.array([18.2, 22.2, 17.4])+273.15, 'K')
    temperatures.name = 'temperature'
    dewpoints = vert_array(np.array([19., 19., 14.3])+273.15, 'K')
    dewpoints.name = 'dewpoint'
    
    ret = parcel.most_unstable_parcel(dat=xarray.merge([levels, temperatures, dewpoints]),
                                      depth=100)
    
    assert_almost_equal(ret.pressure, 959.0, 6)
    assert_almost_equal(ret.temperature, 22.2+273.15, 6)
    assert_almost_equal(ret.dewpoint, 19.0+273.15, 6)

def test_surface_based_cape_cin():
    """Test the surface-based CAPE and CIN calculation."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -38.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
    
    cape_cin = parcel.surface_based_cape_cin(pressure=levels, temperature=temperatures, 
                                             dewpoint=dewpoints)

    assert_almost_equal(cape_cin.cape, 75.0535446, 2)
    assert_almost_equal(cape_cin.cin, -136.685967, 2)

def test_profile_with_nans():
    """Test a profile with nans to make sure it calculates functions appropriately (#1187)."""
    levels = vert_array([1001, 1000, 997, 977.9, 977, 957, 937.8, 925, 906, 899.3, 887, 862.5,
                         854, 850, 800, 793.9, 785, 777, 771, 762, 731.8, 726, 703, 700, 655,
                         630, 621.2, 602, 570.7, 548, 546.8, 539, 513, 511, 485, 481, 468,
                         448, 439, 424, 420, 412], 'hPa')
    levels.name = 'pressure'
    temperatures = vert_array(np.array([-22.5, -22.7, -23.1, np.nan, -24.5, -25.1, np.nan, -24.5, -23.9,
                                        np.nan, -24.7, np.nan, -21.3, -21.3, -22.7, np.nan, -20.7, -16.3,
                                        -15.5, np.nan, np.nan, -15.3, np.nan, -17.3, -20.9, -22.5,
                                        np.nan, -25.5, np.nan, -31.5, np.nan, -31.5, -34.1, -34.3,
                                        -37.3, -37.7, -39.5, -42.1, -43.1, -45.1, -45.7, -46.7])+273.15, 'K')
    temperatures.name = 'temperature'
    dewpoints = vert_array(np.array([-25.1, -26.1, -26.8, np.nan, -27.3, -28.2, np.nan, -27.2, -26.6,
                                     np.nan, -27.4, np.nan, -23.5, -23.5, -25.1, np.nan, -22.9, -17.8,
                                     -16.6, np.nan, np.nan, -16.4, np.nan, -18.5, -21, -23.7, np.nan,
                                     -28.3, np.nan, -32.6, np.nan, -33.8, -35, -35.1, -38.1, -40,
                                     -43.3, -44.6, -46.4, -47, -49.2, -50.7])+273.15, 'K')
    dewpoints.name = 'dewpoint'
    
    # Calculate parcel profile without LCL, as per metpy unit tests.
    profile = parcel.parcel_profile(pressure=levels, 
                                    parcel_pressure=levels[0], 
                                    parcel_temperature=temperatures[0],
                                    parcel_dewpoint=dewpoints[0])
    profile['environment_temperature'] = temperatures
    lfc = parcel.lfc_el(profile=profile)
        
    cape_cin_base = parcel.cape_cin_base(pressure=levels, 
                                    temperature=temperatures,
                                    lfc_pressure=lfc.lfc_pressure,
                                    el_pressure=lfc.el_pressure,
                                    parcel_profile=profile)
    cape_cin_surf = parcel.surface_based_cape_cin(pressure=levels, 
                                                  temperature=temperatures, 
                                                  dewpoint=dewpoints)
    cape_cin_unstable = parcel.most_unstable_cape_cin(pressure=levels, 
                                                      temperature=temperatures, 
                                                      dewpoint=dewpoints)
    
    assert(np.isnan(lfc.lfc_pressure))
    assert_almost_equal(cape_cin_base.cape, 0, 0)
    assert_almost_equal(cape_cin_base.cin, 0, 0)
    assert_almost_equal(cape_cin_surf.cape, 0, 0)
    assert_almost_equal(cape_cin_surf.cin, 0, 0)
    assert_almost_equal(cape_cin_unstable.cape, 0, 0)
    assert_almost_equal(cape_cin_unstable.cin, 0, 0)

def test_most_unstable_cape_cin_surface():
    """Test the most unstable CAPE/CIN calculation when surface is most unstable."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -38.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
    levels.name = 'pressure'
    temperatures.name = 'temperature'
    dewpoints.name = 'dewpoint'
    
    cape_cin = parcel.most_unstable_cape_cin(pressure=levels, temperature=temperatures, 
                                             dewpoint=dewpoints)
    
    assert_almost_equal(cape_cin.cape, 75.0535446, 2)
    assert_almost_equal(cape_cin.cin, -136.685967, 2)

def test_most_unstable_cape_cin():
    """Test the most unstable CAPE/CIN calculation."""
    pressure = np.array([1000., 959., 867.9, 850., 825., 800.]) * units.mbar
    temperature = np.array([18.2, 22.2, 17.4, 10., 0., 15]) * units.celsius
    dewpoint = np.array([19., 19., 14.3, 0., -10., 0.]) * units.celsius
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    assert_almost_equal(mucape, 157.11404 * units('joule / kilogram'), 4)
    assert_almost_equal(mucin, -31.8406578 * units('joule / kilogram'), 4)

def test_mixed_parcel():
    """Test the mixed parcel calculation."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    levels.name = 'pressure'
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -38.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
    
    mixed = parcel.mixed_parcel(pressure=levels, temperature=temperatures, dewpoint=dewpoints,
                                depth=250)
    assert_almost_equal(mixed.pressure, 959., 6)
    assert_almost_equal(mixed.temperature, 28.7401463+273.15, 6) 
    assert_almost_equal(mixed.dewpoint, 7.1534658+273.15, 6)

def test_mixed_layer_cape_cin():
    """Test the calculation of mixed layer cape/cin."""
    levels, temperatures, dewpoints = multiple_intersections()
    
    cape_cin = parcel.mixed_layer_cape_cin(pressure=levels, temperature=temperatures, 
                                           dewpoint=dewpoints)
    
    assert_almost_equal(cape_cin.cape, 987.7323, 2)
    assert_almost_equal(cape_cin.cin, -20.6727628, 2)

def test_mixed_layer():
    """Test the mixed layer calculation."""
    pressure = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    pressure.name = 'pressure'
    temperature = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -38.])+273.15, 'K')
    temperature.name = 'temperature'
    mixed = parcel.mixed_layer(xarray.merge([pressure, temperature]), depth=250)
    assert_almost_equal(mixed.temperature, 16.4024930+273.15, 6)

def test_lfc_not_below_lcl():
    """Test sounding where LFC appears to be (but isn't) below LCL."""
    levels = vert_array([1002.5, 1001.7, 1001., 1000.3, 999.7, 999., 998.2, 977.9,
                         966.2, 952.3, 940.6, 930.5, 919.8, 909.1, 898.9, 888.4,
                         878.3, 868.1, 858., 848., 837.2, 827., 816.7, 805.4], 'hPa')
    temperatures = vert_array(np.array([17.9, 17.9, 17.8, 17.7, 17.7, 17.6, 17.5, 16.,
                                        15.2, 14.5, 13.8, 13., 12.5, 11.9, 11.4, 11.,
                                        10.3, 9.7, 9.2, 8.7, 8., 7.4, 6.8, 6.1])+273.15, 'K')
    dewpoints = vert_array(np.array([13.6, 13.6, 13.5, 13.5, 13.5, 13.5, 13.4, 12.5,
                                     12.1, 11.8, 11.4, 11.3, 11., 9.3, 10., 8.7, 8.9,
                                     8.6, 8.1, 7.6, 7., 6.5, 6., 5.4])+273.15, 'K')
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc_el = parcel.lfc_el(profile=profile)
    assert_almost_equal(lfc_el.lfc_pressure, 811.618879, 3)
    assert_almost_equal(lfc_el.lfc_temperature, 6.48644650+273.15, 3)

def multiple_intersections():
    """Create profile with multiple LFCs and ELs for testing."""
    levels = vert_array([966., 937.2, 925., 904.6, 872.6, 853., 850., 836., 821., 811.6, 782.3,
                         754.2, 726.9, 700., 648.9, 624.6, 601.1, 595., 587., 576., 555.7,
                         534.2, 524., 500., 473.3, 400., 384.5, 358., 343., 308.3, 300., 276.,
                         273., 268.5, 250., 244.2, 233., 200.], 'hPa')
    levels.name = 'pressure'
    
    temperatures = vert_array(np.array([18.2, 16.8, 16.2, 15.1, 13.3, 12.2, 12.4, 14., 14.4,
                                        13.7, 11.4, 9.1, 6.8, 4.4, -1.4, -4.4, -7.3, -8.1,
                                        -7.9, -7.7, -8.7, -9.8, -10.3, -13.5, -17.1, -28.1, -30.7,
                                        -35.3, -37.1, -43.5, -45.1, -49.9, -50.4, -51.1, -54.1, -55.,
                                        -56.7, -57.5])+273.15, 'K')
    temperatures.name = 'temperature'
    
    dewpoints = vert_array(np.array([16.9, 15.9, 15.5, 14.2, 12.1, 10.8, 8.6, 0., -3.6, -4.4,
                                     -6.9, -9.5, -12., -14.6, -15.8, -16.4, -16.9, -17.1, -27.9, -42.7,
                                     -44.1, -45.6, -46.3, -45.5, -47.1, -52.1, -50.4, -47.3, -57.1,
                                     -57.9, -58.1, -60.9, -61.4, -62.1, -65.1, -65.6,
                                     -66.7, -70.5])+273.15, 'K')
    dewpoints.name = 'dewpoint'
    
    return levels, temperatures, dewpoints

def test_multiple_lfcs_el_simple():
    """Test sounding with multiple LFCs."""
    levels, temperatures, dewpoints = multiple_intersections()
    
    profile = parcel.parcel_profile_with_lcl(pressure=levels, 
                                             temperature=temperatures,
                                             parcel_pressure=levels[0], 
                                             parcel_temperature=temperatures[0],
                                             parcel_dewpoint=dewpoints[0])
    lfc_el = parcel.lfc_el(profile=profile)
    
    assert_almost_equal(lfc_el.lfc_pressure, 884.14790, 3)
    assert_almost_equal(lfc_el.lfc_temperature, 13.95707016+273.15, 3)
    assert_almost_equal(lfc_el.el_pressure, 228.151466, 3)
    assert_almost_equal(lfc_el.el_temperature, -56.81015490+273.15, 3)

def test_cape_cin_custom_profile():
    """Test the CAPE and CIN calculation with a custom profile passed to LFC and EL."""
    levels = vert_array([959., 779.2, 751.3, 724.3, 700., 269.], 'hPa')
    temperatures = vert_array(np.array([22.2, 14.6, 12., 9.4, 7., -38.])+273.15, 'K')
    dewpoints = vert_array(np.array([19., -11.2, -10.8, -10.4, -10., -53.2])+273.15, 'K')
   
    profile = parcel.parcel_profile(pressure=levels, 
                                    parcel_pressure=levels[0], 
                                    parcel_temperature=temperatures[0],
                                    parcel_dewpoint=dewpoints[0])
    profile['temperature'] = profile.temperature + 5
    profile['environment_temperature'] = temperatures
    
    lfc = parcel.lfc_el(profile=profile)
    cape_cin = parcel.cape_cin_base(pressure=levels, 
                                    temperature=temperatures,
                                    lfc_pressure=lfc.lfc_pressure,
                                    el_pressure=lfc.el_pressure,
                                    parcel_profile=profile)

    assert_almost_equal(cape_cin.cape, 1440.463208696, 2)
    assert_almost_equal(cape_cin.cin, 0.0, 2)

def test_parcel_profile_below_lcl():
    """Test parcel profile calculation when pressures do not reach LCL (#827)."""
    pressure = vert_array([981, 949.2, 925., 913.9, 903, 879.4, 878, 864, 855,
                           850, 846.3, 838, 820, 814.5, 799, 794], 'hPa')
    truth = np.array([276.35, 273.760341, 271.747753, 270.812026, 269.885225,
                      267.850849, 267.728946, 266.502214, 265.706084, 265.261201,
                      264.930782, 264.185801, 262.551884, 262.047526, 260.61294,
                      260.145932])
    
    parcel_temperature = xarray.DataArray(3.2+273.15, attrs={'units': 'K'})
    parcel_dewpoint = xarray.DataArray(-10.8+273.15, attrs={'units': 'K'})
    
    profile = parcel.parcel_profile(pressure=pressure, 
                                    parcel_pressure=pressure[0], 
                                    parcel_temperature=parcel_temperature,
                                    parcel_dewpoint=parcel_dewpoint)
    
    assert_array_almost_equal(profile.temperature, truth, 6)

def test_lcl_convergence_issue():
    """Test profile where LCL wouldn't converge (#1187)."""
    pressure = vert_array([990, 973, 931, 925, 905], 'hPa')
    temperatures = vert_array(np.array([14.4, 14.2, 13, 12.6, 11.4])+273.15, 'K')
    dewpoints = vert_array(np.array([14.4, 11.7, 8.2, 7.8, 7.6])+273.15, 'K')
    
    lcl = parcel.lcl(parcel_pressure=pressure[0], 
                     parcel_temperature=temperatures[0], 
                     parcel_dewpoint=dewpoints[0])
    assert_almost_equal(lcl.lcl_pressure, 990, 0)

def test_cape_cin_value_error():
    """Test a profile that originally caused a ValueError in #1190."""
    levels = vert_array([1012.0, 1009.0, 1002.0, 1000.0, 925.0, 896.0, 855.0, 850.0, 849.0,
                         830.0, 775.0, 769.0, 758.0, 747.0, 741.0, 731.0, 712.0, 700.0, 691.0,
                         671.0, 636.0, 620.0, 610.0, 601.0, 594.0, 587.0, 583.0, 580.0, 571.0,
                         569.0, 554.0, 530.0, 514.0, 506.0, 502.0, 500.0, 492.0, 484.0, 475.0,
                         456.0, 449.0, 442.0, 433.0, 427.0, 400.0, 395.0, 390.0, 351.0, 300.0,
                         298.0, 294.0, 274.0, 250.0], 'hPa')
    temperatures = vert_array(np.array([27.8, 25.8, 24.2, 24, 18.8, 16, 13, 12.6, 12.6, 11.6, 9.2, 8.6,
                                        8.4, 9.2, 10, 9.4, 7.4, 6.2, 5.2, 3.2, -0.3, -2.3, -3.3, -4.5,
                                        -5.5, -6.1, -6.1, -6.1, -6.3, -6.3, -7.7, -9.5, -9.9, -10.3,
                                        -10.9, -11.1, -11.9, -12.7, -13.7, -16.1, -16.9, -17.9, -19.1,
                                        -19.9, -23.9, -24.7, -25.3, -29.5, -39.3, -39.7, -40.5, -44.3,
                                        -49.3])+273.15, 'K')
    dewpoints = vert_array(np.array([19.8, 16.8, 16.2, 16, 13.8, 12.8, 10.1, 9.7, 9.7,
                                     8.6, 4.2, 3.9, 0.4, -5.8, -32, -34.6, -35.6, -34.8,
                                     -32.8, -10.8, -9.3, -10.3, -9.3, -10.5, -10.5, -10, -16.1,
                                     -19.1, -23.3, -18.3, -17.7, -20.5, -27.9, -32.3, -33.9, -34.1,
                                     -35.9, -26.7, -37.7, -43.1, -33.9, -40.9, -46.1, -34.9, -33.9,
                                     -33.7, -33.3, -42.5, -50.3, -49.7, -49.5, -58.3, -61.3])+273.15, 'K')
    
    cape_cin = parcel.surface_based_cape_cin(pressure=levels, temperature=temperatures, 
                                             dewpoint=dewpoints)
    
    assert_almost_equal(cape_cin.cape, 2007.040698, 3)
    assert_almost_equal(cape_cin.cin, 0.0, 3)

def test_lcl_grid_surface_lcls():
    """Test surface grid where some values have LCLs at the surface."""
    pressure = vert_array([1000, 990, 1010], 'hPa').expand_dims({'x': 3})
    temperature = vert_array(np.array([15, 14, 13])+273.15, 'K').expand_dims({'x': 3})
    dewpoint = vert_array(np.array([15, 10, 13])+273.15, 'K').expand_dims({'x': 3})

    lcl = parcel.lcl(parcel_pressure=pressure[0], 
                     parcel_temperature=temperature[0], 
                     parcel_dewpoint=dewpoint[0])
    
    pres_truth = np.array([1000, 932.1719, 1010]) 
    temp_truth = np.array([15, 9.10424, 13])+273.15
    assert_array_almost_equal(lcl.lcl_pressure, pres_truth, 4)
    assert_array_almost_equal(lcl.lcl_temperature, temp_truth, 4)

def test_lifted_index():
    """Test the Lifted Index calculation."""
    pressure = vert_array([1014., 1000., 997., 981.2, 947.4, 925., 914.9, 911.,
                           902., 883., 850., 822.3, 816., 807., 793.2, 770.,
                           765.1, 753., 737.5, 737., 713., 700., 688., 685.,
                           680., 666., 659.8, 653., 643., 634., 615., 611.8,
                           566.2, 516., 500., 487., 484.2, 481., 475., 460.,
                           400.], 'hPa')
    pressure.name = 'pressure'
    
    temperature = vert_array(np.array([24.2, 24.2, 24., 23.1, 21., 19.6, 18.7, 18.4,
                                       19.2, 19.4, 17.2, 15.3, 14.8, 14.4, 13.4, 11.6,
                                       11.1, 10., 8.8, 8.8, 8.2, 7., 5.6, 5.6,
                                       5.6, 4.4, 3.8, 3.2, 3., 3.2, 1.8, 1.5,
                                       -3.4, -9.3, -11.3, -13.1, -13.1, -13.1, -13.7, -15.1,
                                       -23.5])+273.15, 'K')
    temperature.name = 'temperature'
    
    dewpoint = vert_array(np.array([23.2, 23.1, 22.8, 22., 20.2, 19., 17.6, 17.,
                                    16.8, 15.5, 14., 11.7, 11.2, 8.4, 7., 4.6,
                                    5., 6., 4.2, 4.1, -1.8, -2., -1.4, -0.4,
                                    -3.4, -5.6, -4.3, -2.8, -7., -25.8, -31.2, -31.4,
                                    -34.1, -37.3, -32.3, -34.1, -37.3, -41.1, -37.7, -58.1,
                                    -57.5])+273.15, 'K')
    
    # Use profile without lcl as per metpy unit test.
    profile = parcel.parcel_profile(pressure=pressure, 
                                    parcel_pressure=pressure[0], 
                                    parcel_temperature=temperature[0],
                                    parcel_dewpoint=dewpoint[0])
    profile['environment_temperature'] = temperature
    
    li = parcel.lifted_index(profile=profile)
    assert_almost_equal(li.lifted_index, -7.9176350, 2)
