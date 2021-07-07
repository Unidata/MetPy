# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Cartopy specific mapping utilities."""

try:
    from cartopy.feature import Feature, Scaler

    from ..cbook import get_test_data

    class MetPyMapFeature(Feature):
        """A simple interface to MetPy-included shapefiles."""

        def __init__(self, name, scale, **kwargs):
            """Create MetPyMapFeature instance."""
            import cartopy.crs as ccrs
            super().__init__(ccrs.PlateCarree(), **kwargs)
            self.name = name

            if isinstance(scale, str):
                scale = Scaler(scale)
            self.scaler = scale

        def geometries(self):
            """Return an iterator of (shapely) geometries for this feature."""
            import cartopy.io.shapereader as shapereader

            # Ensure that the associated files are in the cache
            fname = f'{self.name}_{self.scaler.scale}'
            for extension in ['.dbf', '.shx']:
                get_test_data(fname + extension, as_file_obj=False)
            path = get_test_data(fname + '.shp', as_file_obj=False)
            return iter(tuple(shapereader.Reader(path).geometries()))

        def intersecting_geometries(self, extent):
            """Return geometries that intersect the extent."""
            self.scaler.scale_from_extent(extent)
            return super().intersecting_geometries(extent)

        def with_scale(self, new_scale):
            """
            Return a copy of the feature with a new scale.

            Parameters
            ----------
            new_scale
                The new dataset scale, i.e. one of '500k', '5m', or '20m'.
                Corresponding to 1:500,000, 1:5,000,000, and 1:20,000,000
                respectively.

            """
            return MetPyMapFeature(self.name, new_scale, **self.kwargs)

    USCOUNTIES = MetPyMapFeature('us_counties', '20m', facecolor='None', edgecolor='black')

    USSTATES = MetPyMapFeature('us_states', '20m', facecolor='None', edgecolor='black')
except ImportError:
    # If no Cartopy is present, we just don't have map features
    pass

__all__ = ['USCOUNTIES', 'USSTATES']


def import_cartopy():
    """Import CartoPy; return a stub if unable.

    This allows code requiring CartoPy to fail at use time rather than import time.
    """
    try:
        import cartopy.crs as ccrs
        return ccrs
    except ImportError:
        return CartopyStub()


class CartopyStub:
    """Fail if a CartoPy attribute is accessed."""

    def __getattr__(self, name):
        """Raise an error on any attribute access."""
        raise AttributeError(f'Cannot use {name} without Cartopy installed.')
