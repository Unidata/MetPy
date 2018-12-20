#  Copyright (c) 2018,2019 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Declarative plotting tools."""

from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from traitlets import (Any, Bool, Float, HasTraits, Instance, Int, List, observe, Tuple,
                       Unicode, Union)

from . import cartopy_utils, ctables
from ..cbook import is_string_like
from ..package_tools import Exporter
from ..units import units

exporter = Exporter(globals())

_projections = {'lcc': ccrs.LambertConformal(central_latitude=40, central_longitude=-100,
                                             standard_parallels=[30, 60]),
                'ps': ccrs.NorthPolarStereo(central_longitude=-100),
                'mer': ccrs.Mercator()}

_areas = {'us': (-115, -65, 25, 50),
          'hi': (-161.5, -152.5, 17, 23)}


class Panel(HasTraits):
    """Draw one or more plots."""


@exporter.export
class PanelContainer(HasTraits):
    """Hold multiple panels of plots."""

    size = Union([Tuple(Int(), Int()), Instance(type(None))], default_value=None)
    panels = List(Instance(Panel))

    @property
    def panel(self):
        """Provide simple access for a single panel."""
        return self.panels[0]

    @panel.setter
    def panel(self, val):
        self.panels = [val]

    @observe('panels')
    def _panels_changed(self, change):
        for panel in change.new:
            panel.parent = self
            panel.observe(self.refresh, names=('_need_redraw'))

    @property
    def figure(self):
        """Provide access to the underlying figure object."""
        if not hasattr(self, '_fig'):
            self._fig = plt.figure(figsize=self.size)
        return self._fig

    def refresh(self, _):
        """Refresh the rendering of all panels."""
        # First make sure everything is properly constructed
        self.draw()

        # Trigger the graphics refresh
        self.figure.canvas.draw()

        # Flush out interactive events--only ok on Agg for newer matplotlib
        try:
            self.figure.canvas.flush_events()
        except NotImplementedError:
            pass

    def draw(self):
        """Draw the collection of panels."""
        for panel in self.panels:
            with panel.hold_trait_notifications():
                panel.draw()

    def save(self, *args, **kwargs):
        """Save the constructed graphic as an image file."""
        self.draw()
        self.figure.savefig(*args, **kwargs)

    def show(self):
        """Show the constructed graphic on the screen."""
        self.draw()
        self.figure.show()


@exporter.export
class MapPanel(Panel):
    """Draw one or more plots on a map."""

    parent = Instance(PanelContainer)
    layout = Tuple(Int(), Int(), Int(), default_value=(1, 1, 1))
    plots = List(Any())
    _need_redraw = Bool(default_value=True)

    area = Union([Unicode(), Tuple(Float(), Float(), Float(), Float())], allow_none=True,
                 default_value=None)
    projection = Union([Unicode(), Instance(ccrs.Projection)], default_value='data')
    maps = List(Union([Unicode(), Instance(cfeature.Feature)]), default_value=['coastline'])
    title = Unicode()

    @observe('plots')
    def _plots_changed(self, change):
        """Handle when our collection of plots changes."""
        for plot in change.new:
            plot.parent = self
            plot.observe(self.refresh, names=('_need_redraw'))
        self._need_redraw = True

    @observe('parent')
    def _parent_changed(self, _):
        """Handle when the parent is changed."""
        self.ax = None

    @property
    def _proj_obj(self):
        """Return the projection as a Cartopy object.

        Handles looking up a string for the projection, or if the projection
        is set to ``'data'`` looks at the data for the projection.

        """
        if is_string_like(self.projection):
            if self.projection == 'data':
                return self.plots[0].griddata.metpy.cartopy_crs
            else:
                return _projections[self.projection]
        else:
            return self.projection

    @property
    def _layer_features(self):
        """Iterate over all map features and return as Cartopy objects.

        Handle converting names of maps to auto-scaling map features.

        """
        for item in self.maps:
            if is_string_like(item):
                item = item.upper()
                try:
                    scaler = cfeature.AdaptiveScaler('110m', (('50m', 50), ('10m', 15)))
                    feat = getattr(cfeature, item).with_scale(scaler)
                except AttributeError:
                    scaler = cfeature.AdaptiveScaler('20m', (('5m', 5), ('500k', 1)))
                    feat = getattr(cartopy_utils, item).with_scale(scaler)
            else:
                feat = item

            yield feat

    @observe('area')
    def _set_need_redraw(self, _):
        """Watch traits and set the need redraw flag as necessary."""
        self._need_redraw = True

    @property
    def ax(self):
        """Get the :class:`matplotlib.axes.Axes` to draw on.

        Creates a new instance if necessary.

        """
        # If we haven't actually made an instance yet, make one with the right size and
        # map projection.
        if getattr(self, '_ax', None) is None:
            self._ax = self.parent.figure.add_subplot(*self.layout, projection=self._proj_obj)

        return self._ax

    @ax.setter
    def ax(self, val):
        """Set the :class:`matplotlib.axes.Axes` to draw on.

        Clears existing state as necessary.

        """
        if getattr(self, '_ax', None) is not None:
            self._ax.cla()
        self._ax = val

    def refresh(self, changed):
        """Refresh the drawing if necessary."""
        self._need_redraw = changed.new

    def draw(self):
        """Draw the panel."""
        # Only need to run if we've actually changed.
        if self._need_redraw:

            # Draw all of the plots.
            for p in self.plots:
                with p.hold_trait_notifications():
                    p.draw()

            # Add all of the maps
            for feat in self._layer_features:
                self.ax.add_feature(feat)

            # Set the extent as appropriate based on the area. One special case for 'global'
            if self.area == 'global':
                self.ax.set_global()
            elif self.area is not None:
                # Try to look up if we have a string
                if is_string_like(self.area):
                    area = _areas[self.area]
                # Otherwise, assume we have a tuple to use as the extent
                else:
                    area = self.area
                self.ax.set_extent(area, ccrs.PlateCarree())

            # Use the set title or generate one.
            title = self.title or ', '.join(plot.name for plot in self.plots)
            self.ax.set_title(title)
            self._need_redraw = False


@exporter.export
class Plot2D(HasTraits):
    """Represent plots of 2D data."""

    parent = Instance(Panel)
    _need_redraw = Bool(default_value=True)

    field = Unicode()
    level = Union([Int(allow_none=True, default_value=None), Instance(units.Quantity)])
    time = Instance(datetime, allow_none=True)

    colormap = Unicode(allow_none=True, default_value=None)
    image_range = Union([Tuple(Int(allow_none=True), Int(allow_none=True)),
                         Instance(plt.Normalize)], default_value=(None, None))

    @property
    def _cmap_obj(self):
        """Return the colormap object.

        Handle convert the name of the colormap to an object from matplotlib or metpy.

        """
        try:
            return ctables.registry.get_colortable(self.colormap)
        except KeyError:
            return plt.get_cmap(self.colormap)

    @property
    def _norm_obj(self):
        """Return the normalization object.

        Converts the tuple image range to a matplotlib normalization instance.

        """
        return plt.Normalize(*self.image_range)

    def clear(self):
        """Clear the plot.

        Resets all internal state and sets need for redraw.

        """
        if getattr(self, 'handle', None) is not None:
            self.clear_handle()
            self._need_redraw = True

    def clear_handle(self):
        """Clear the handle to the plot instance."""
        self.handle.remove()
        self.handle = None

    @observe('parent')
    def _parent_changed(self, _):
        """Handle setting the parent object for the plot."""
        self.clear()

    @observe('field', 'level', 'time')
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._griddata = None
        self.clear()

    # Can't be a Traitlet because notifications don't work with arrays for traits
    # notification never happens
    @property
    def data(self):
        """Access the current data subset."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        self._update_data()

    @property
    def griddata(self):
        """Return the internal cached data."""
        if getattr(self, '_griddata', None) is None:

            if self.field:
                data = self.data.metpy.parse_cf(self.field)
            elif not hasattr(self.data.metpy, 'x'):
                # Handles the case where we have a dataset but no specified field
                raise ValueError('field attribute has not been set.')
            else:
                data = self.data

            subset = {'method': 'nearest'}
            if self.level is not None:
                subset[data.metpy.vertical.name] = self.level

            if self.time is not None:
                subset[data.metpy.time.name] = self.time
            self._griddata = data.metpy.sel(**subset).squeeze()

        return self._griddata

    @property
    def plotdata(self):
        """Return the data for plotting.

        The data array, x coordinates, and y coordinates.

        """
        x = self.griddata.metpy.x
        y = self.griddata.metpy.y

        if 'degree' in x.units:
            import numpy as np
            x, y, _ = self.griddata.metpy.cartopy_crs.transform_points(ccrs.PlateCarree(),
                                                                       *np.meshgrid(x, y)).T
            x = x[:, 0] % 360
            y = y[0, :]

        return x, y, self.griddata

    @property
    def name(self):
        """Generate a name for the plot."""
        ret = self.field
        if self.level is not None:
            ret += '@{:d}'.format(self.level)
        return ret

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, 'handle', None) is None:
                self._build()
            self._need_redraw = False


@exporter.export
class ImagePlot(Plot2D):
    """Represent an image plot."""

    @observe('colormap', 'image_range')
    def _set_need_redraw(self, _):
        """Handle changes to attributes that just need a simple redraw."""
        if hasattr(self, 'handle'):
            self.handle.set_cmap(self._cmap_obj)
            self.handle.set_norm(self._norm_obj)
            self._need_redraw = True

    @property
    def plotdata(self):
        """Return the data for plotting.

        The data array, x coordinates, and y coordinates.

        """
        x = self.griddata.metpy.x
        y = self.griddata.metpy.y

        # At least currently imshow with cartopy does not like this
        if 'degree' in x.units:
            x = x.data
            x[x > 180] -= 360

        return x, y, self.griddata

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x, y, imdata = self.plotdata

        # We use min/max for y and manually figure out origin to try to avoid upside down
        # images created by images where y[0] > y[-1]
        extents = (x[0], x[-1], y.min(), y.max())
        origin = 'upper' if y[0] > y[-1] else 'lower'
        self.handle = self.parent.ax.imshow(imdata, extent=extents, origin=origin,
                                            cmap=self._cmap_obj, norm=self._norm_obj,
                                            transform=imdata.metpy.cartopy_crs)


@exporter.export
class ContourPlot(Plot2D):
    """Represent a contour plot."""

    linecolor = Unicode('black')
    linewidth = Int(2)
    contours = Union([List(Float()), Int()], default_value=25)

    @observe('contours', 'linecolor', 'linewidth')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def clear_handle(self):
        """Clear the handle to the plot instance."""
        for col in self.handle.collections:
            col.remove()
        self.handle = None

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x, y, imdata = self.plotdata
        self.handle = self.parent.ax.contour(x, y, imdata, self.contours,
                                             colors=self.linecolor, linewidths=self.linewidth,
                                             transform=imdata.metpy.cartopy_crs)
