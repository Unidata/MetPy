# Copyright (c) 2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Cartopy specific mapping utilities."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ..cbook import get_test_data


class MetPyMapFeature(cfeature.NaturalEarthFeature):
    """A simple interface to US County shapefiles."""

    def __init__(self, name, scale, **kwargs):
        """Create USCountiesFeature instance."""
        super(cfeature.NaturalEarthFeature, self).__init__(ccrs.PlateCarree(), **kwargs)

        self.category = ''
        self.name = name
        self.scaler = (
            cfeature.Scaler(scale) if not isinstance(scale, cfeature.Scaler) else scale
        )

    def geometries(self):
        """Return an iterator of (shapely) geometries for this feature."""
        import cartopy.io.shapereader as shapereader
        # Ensure that the associated files are in the cache
        fname = '{}_{}'.format(self.name, self.scale)
        for extension in ['.dbf', '.shx']:
            get_test_data(fname + extension)
        path = get_test_data(fname + '.shp', as_file_obj=False)
        return iter(tuple(shapereader.Reader(path).geometries()))

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
