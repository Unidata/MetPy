=======================
MetPy 1.0 Upgrade Guide
=======================

.. toctree::
   :maxdepth: 1
   :hidden:

   apichange

The 1.0 release marks the first release of MetPy to have what we call a "stable" interface.
This means that code that works with a given release of MetPy 1 should work with all
later versions of MetPy 1. Given those constraints, many of MetPy's functions changed in
some way in MetPy 1.0. In the majority of these cases, the change was to rename some of
the function parameters for consistency (which only affects users passing in these parameters
as keyword arguments). For more details see the full list of :doc:`apichange`.

The biggest change in MetPy 1.0 is expanded support for xarray ``DataArray`` instances as
input to MetPy calculation functions. MetPy calculations can now return a ``DataArray`` when
given such as input, and can also take advantage of the metadata available on a ``DataArray``
(such as coordinate information) in order to simplify code. For example, calculating
the geostrophic wind used to look like:

.. code-block:: python

    # Read data and get the geopotential heights for a single time and level
    ds = xr.open_dataset(get_test_data('irma_gfs_example.nc', as_file_obj=False))
    height = ds.metpy.parse_cf('Geopotential_height_isobaric').isel(time1=0, isobaric3=0)

    # All the calculations needed for geostrophic wind
    dx, dy = mpcalc.lat_lon_grid_deltas(height.longitude.metpy.unit_array,
                                        height.latitude.metpy.unit_array)
    f = mpcalc.coriolis_parameter(height.latitude.metpy.unit_array)
    ug, vg = mpcalc.geostrophic_wind(height.metpy.unit_array, f[:, None], dx, dy)

These last three lines (of calculation) now become:

.. code-block:: python

    ug, vg = mpcalc.geostrophic_wind(height)

For more information on how to best use xarray, see the :doc:`/tutorials/xarray_tutorial`.

In 1.0, we have also expanded our declarative plotting interface with more capabilities. For
more information, see the :doc:`/tutorials/declarative_tutorial`.
