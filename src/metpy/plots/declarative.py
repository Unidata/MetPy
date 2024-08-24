#  Copyright (c) 2018,2019 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Declarative plotting tools."""
import collections
import contextlib
import copy
from datetime import datetime, timedelta
from difflib import get_close_matches
from itertools import cycle
import re

from matplotlib.contour import ContourSet
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from traitlets import (Any, Bool, Dict, Float, HasTraits, Instance, Int, List, observe,
                       TraitError, Tuple, Unicode, Union, validate)

from . import ctables, wx_symbols
from ._mpl import TextCollection
from .cartopy_utils import import_cartopy
from .station_plot import StationPlot
from ..calc import reduce_point_density, smooth_n_point, zoom_xarray
from ..package_tools import Exporter
from ..units import units

ccrs = import_cartopy()
exporter = Exporter(globals())


def lookup_projection(projection_code):
    """Get a Cartopy projection based on a short abbreviation."""
    import cartopy.crs as ccrs

    projections = {'lcc': ccrs.LambertConformal(central_latitude=40, central_longitude=-100,
                                                standard_parallels=[30, 60]),
                   'ps': ccrs.NorthPolarStereo(central_longitude=-100),
                   'mer': ccrs.Mercator()}
    return projections[projection_code]


def lookup_map_feature(feature_name):
    """Get a Cartopy map feature based on a name."""
    import cartopy.feature as cfeature

    from . import cartopy_utils

    name = feature_name.upper()
    try:
        feat = getattr(cfeature, name)
        scaler = cfeature.AdaptiveScaler('110m', (('50m', 50), ('10m', 15)))
    except AttributeError:
        feat = getattr(cartopy_utils, name)
        scaler = cfeature.AdaptiveScaler('20m', (('5m', 5), ('500k', 1)))
    return feat.with_scale(scaler)


def plot_kwargs(data, args):
    """Set the keyword arguments for MapPanel plotting."""
    if hasattr(data.metpy, 'cartopy_crs'):
        # Conditionally add cartopy transform if we are on a map.
        kwargs = {'transform': data.metpy.cartopy_crs}
    else:
        kwargs = {}
    kwargs.update(args)
    return kwargs


def get_cartopy_color(val):
    """Provide the special map feature colors from Cartopy."""
    from cartopy.feature import COLORS
    return COLORS[val]


class ValidationMixin:
    """Provides validation of attribute names when set by user."""

    def __setattr__(self, name, value):
        """Set only permitted attributes."""
        allowlist = ['ax',
                     'data',
                     'handle',
                     'notify_change',
                     'panel'
                     ]

        allowlist.extend(self.trait_names())
        if name in allowlist or name.startswith('_'):
            super().__setattr__(name, value)
        else:
            closest = get_close_matches(name, allowlist, n=1)
            if closest:
                alt = closest[0]
                suggest = f" Perhaps you meant '{alt}'?"
            else:
                suggest = ''
            obj = self.__class__
            msg = f"'{name}' is not a valid attribute for {obj}." + suggest
            raise AttributeError(msg)


class MetPyHasTraits(HasTraits):
    """Provides modification layer on HasTraits for declarative classes."""

    def __dir__(self):
        """Filter dir to be more helpful for tab-completion in Jupyter."""
        return filter(
            lambda name: not (name in dir(HasTraits) or name.startswith('_')),
            dir(type(self))
        )

    mpl_args = Dict(allow_none=True)
    mpl_args.__doc__ = """Supply a dictionary of valid Matplotlib keyword arguments to modify
    how the plot variable is drawn.

    Using this attribute you must choose the appropriate keyword arguments (kwargs) based on
    what you are plotting (e.g., contours, color-filled contours, image plot, etc.). This is
    available for all plot types (ContourPlot, FilledContourPlot, RasterPlot, ImagePlot,
    BarbPlot, ArrowPlot, PlotGeometry, and PlotObs). For PlotObs, the kwargs are those to
    specify the StationPlot object. NOTE: Setting the mpl_args trait will override
    any other trait that corresponds to a specific kwarg for the particular plot type
    (e.g., linecolor, linewidth).
    """


class Panel(MetPyHasTraits):
    """Draw one or more plots."""


@exporter.export
class PanelContainer(MetPyHasTraits, ValidationMixin):
    """Collects panels and set complete figure related settings (e.g., size)."""

    size = Union([Tuple(Union([Int(), Float()]), Union([Int(), Float()])),
                 Instance(type(None))], default_value=None)
    size.__doc__ = """This trait takes a tuple of (width, height) to set the size of the
    figure.

    This trait defaults to None and will assume the default `matplotlib.pyplot.figure` size.
    """

    panels = List(Instance(Panel))
    panels.__doc__ = """A list of panels to plot on the figure.

    This trait must contain at least one panel to plot on the figure."""

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
        with contextlib.suppress(NotImplementedError):
            self.figure.canvas.flush_events()

    def draw(self):
        """Draw the collection of panels."""
        for panel in self.panels:
            with panel.hold_trait_notifications():
                panel.draw()

    def save(self, *args, **kwargs):
        """Save the constructed graphic as an image file.

        This method takes a string for saved file name. Additionally, the same arguments and
        keyword arguments that `matplotlib.pyplot.savefig` does.
        """
        self.draw()
        self.figure.savefig(*args, **kwargs)

    def show(self):
        """Show the constructed graphic on the screen."""
        self.draw()
        plt.show()

    def copy(self):
        """Return a copy of the panel container."""
        return copy.copy(self)


@exporter.export
class MapPanel(Panel, ValidationMixin):
    """Set figure related elements for an individual panel.

    Parameters that need to be set include collecting all plotting types
    (e.g., contours, wind barbs, etc.) that are desired to be in a given panel.
    Additionally, traits can be set to plot map related features (e.g., coastlines, borders),
    projection, graphics area, and title.
    """

    parent = Instance(PanelContainer, allow_none=True)

    layout = Tuple(Int(), Int(), Int(), default_value=(1, 1, 1))
    layout.__doc__ = """A tuple that contains the description (nrows, ncols, index) of the
    panel position; default value is (1, 1, 1).

    This trait is set to describe the panel position and the default is for a single panel. For
    example, a four-panel plot will have two rows and two columns with the tuple setting for
    the upper-left panel as (2, 2, 1), upper-right as (2, 2, 2), lower-left as (2, 2, 3), and
    lower-right as (2, 2, 4). For more details see the documentation for
    `matplotlib.figure.Figure.add_subplot`.
    """

    plots = List(Any())
    plots.__doc__ = """A list of handles that represent the plots (e.g., `ContourPlot`,
    `FilledContourPlot`, `ImagePlot`) to put on a given panel.

    This trait collects the different plots, including contours and images, that are intended
    for a given panel.
    """

    _need_redraw = Bool(default_value=True)

    area = Union([Unicode(), Tuple(Float(), Float(), Float(), Float())], allow_none=True,
                 default_value=None)
    area.__doc__ = """A tuple or string value that indicates the graphical area of the plot.

    The tuple value corresponds to longitude/latitude box based on the projection of the map
    with the format (west-most longitude, east-most longitude, south-most latitude,
    north-most latitude). This tuple defines a box from the lower-left to the upper-right
    corner.

    This trait can also be set with a string value associated with the named geographic regions
    within MetPy. The tuples associated with the names are based on a PlatteCarree projection.
    For a CONUS region, the following strings can be used: 'us', 'spcus', 'ncus', and 'afus'.
    For regional plots, US postal state abbreviations can be used, such as 'co', 'ny', 'ca',
    et cetera. Providing a '+' or '-' suffix to the string value will zoom in or out,
    respectively. Providing multiple '+' or '-' characters will zoom in or out further.

    """

    projection = Union([Unicode(), Instance('cartopy.crs.Projection')], default_value='data')
    projection.__doc__ = """A string for a pre-defined projection or a Cartopy projection
    object.

    There are three pre-defined projections that can be called with a short name:
    Lambert conformal conic ('lcc'), Mercator ('mer'), or polar-stereographic ('ps'), or 'area'
    to use a default projection based on the string area used. Additionally, this trait can be
    set to a Cartopy projection object.
    """

    layers = List(Union([Unicode(), Instance('cartopy.feature.Feature')]),
                  default_value=['coastline'])
    layers.__doc__ = """A list of strings for a pre-defined feature layer or a Cartopy Feature
    object.

    Like the projection, there are a couple of pre-defined feature layers that can be called
    using a short name. The pre-defined layers are: 'coastline', 'states', 'borders', 'lakes',
    'land', 'ocean', 'rivers', 'usstates', and 'uscounties'. Additionally, this can accept
    Cartopy Feature objects.
    """

    layers_edgecolor = List(Unicode(allow_none=True), default_value=['black'])
    layers_edgecolor.__doc__ = """A list of strings for a pre-defined edgecolor for a layer.

    An option to set a different color for the map layer edge colors. Length of list should
    match that of layers if not using default value. Behavior is to repeat colors if not enough
    provided by user. Use `None` value for 'ocean', 'lakes', 'rivers', and 'land'.
    """

    layers_linewidth = List(Union([Int(), Float()], allow_none=True), default_value=[1])
    layers_linewidth.__doc__ = """A list of values defining the linewidth for a layer.

    An option to set a different linewidth for the layer feature. Length of list should
    match that of layers if not using default value. Behavior is to repeat linewidth if
    not enough provided by user. Use `None` value for 'ocean', 'lakes', 'rivers', and 'land'.
    """

    layers_linestyle = List(Unicode(), default_value=['solid'])
    layers_linestyle.__doc__ = """A list of string values defining the linestyle for a layer or
    None.

    Default is `solid`, which, will use a solid lines for drawing the layer. Behavior is to
    repeat linestyle if not enough provided by user.

    The valid string values are those of Matplotlib which are 'solid', 'dashed', 'dotted', and
    'dashdot', as well as their short codes ('-', '--', '.', '-.'). The object `None`, as
    described above, can also be used. Use `None` value for 'ocean', 'lakes', 'rivers', and
    'land'.
    """

    layers_zorder = List(Union([Int(), Float()], allow_none=True), default_value=[None])
    layers_zorder.__doc__ = """A list of values defining the zorder for a layer.

    An option to set a different zorder for the map layer edge colors. Length of list should
    match that of layers if not using default value. Behavior is to repeat zorder if not enough
    provided by user.
    """

    layers_alpha = List(Union([Int(), Float()], allow_none=True), default_value=[1])
    layers_alpha.__doc__ = """A list of values defining the alpha for a layer.

    An option to set a different alpha for the map layer edge colors. Length of list should
    match that of layers if not using default value. Behavior is to repeat alpha if not enough
    provided by user.
    """

    title = Unicode()
    title.__doc__ = """A string to set a title for the figure.

    This trait sets a user-defined title that will plot at the top center of the figure.
    """

    left_title = Unicode(allow_none=True, default_value=None)
    left_title.__doc__ = """A string to set a title for the figure with the location on the
    top left of the figure.

    This trait sets a user-defined title that will plot at the top left of the figure.
    """

    right_title = Unicode(allow_none=True, default_value=None)
    right_title.__doc__ = """A string to set a title for the figure with the location on the
    top right of the figure.

    This trait sets a user-defined title that will plot at the top right of the figure.
    """

    title_fontsize = Union([Int(), Float(), Unicode()], allow_none=True, default_value=None)
    title_fontsize.__doc__ = """An integer or string value for the font size of the title of
    the figure.

    This trait sets the font size for the title that will plot at the top center of the figure.
    Accepts size in points or relative size. Allowed relative sizes are those of Matplotlib:
    'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    """

    @validate('area')
    def _valid_area(self, proposal):
        """Check that proposed string or tuple is valid and turn string into a tuple extent."""
        from .plot_areas import named_areas

        area = proposal['value']

        # Parse string, check that string is valid, and determine extent based on string
        if isinstance(area, str):
            match = re.match(r'(\w+)([-+]*)$', area)
            if match is None:
                raise TraitError(f'"{area}" is not a valid area.')
            region, modifier = match.groups()
            region = region.lower()

            if region == 'global':
                extent = 'global'
            elif region in named_areas:
                self._area_proj = named_areas[region].projection
                extent = named_areas[region].bounds
                zoom = modifier.count('+') - modifier.count('-')
                extent = self._zoom_extent(extent, zoom)
            else:
                raise TraitError(f'"{area}" is not a valid string area.')
        # Otherwise, assume area is a tuple and check that latitudes/longitudes are valid
        else:
            west_lon, east_lon, south_lat, north_lat = area
            valid_west = -180 <= west_lon <= 180
            valid_east = -180 <= east_lon <= 180
            valid_south = -90 <= south_lat <= 90
            valid_north = -90 <= north_lat <= 90
            if not (valid_west and valid_east and valid_south and valid_north):
                raise TraitError(f'"{area}" is not a valid string area.')
            extent = area

        return extent

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
        if isinstance(self.projection, str):
            if self.projection == 'data':
                if isinstance(self.plots[0].griddata, tuple):
                    proj = self.plots[0].griddata[0].metpy.cartopy_crs
                else:
                    proj = self.plots[0].griddata.metpy.cartopy_crs
            elif self.projection == 'area':
                proj = self._area_proj
            else:
                proj = lookup_projection(self.projection)
        else:
            proj = self.projection
        return proj

    @property
    def _layer_features(self):
        """Iterate over all map features and return as Cartopy objects.

        Handle converting names of maps to auto-scaling map features.

        """
        for item in self.layers:
            feat = lookup_map_feature(item) if isinstance(item, str) else item
            yield feat

    @observe('area')
    def _set_need_redraw(self, _):
        """Watch traits and set the need redraw flag as necessary."""
        self._need_redraw = True

    @staticmethod
    def _zoom_extent(extent, zoom):
        """Calculate new bounds for zooming in or out of a given extent.

        ``extent`` is given as a tuple with four numeric values, in the same format as the
        ``area`` trait.

        If ``zoom`` = 0, the extent will not be changed from what was provided to the method
        If ``zoom`` > 0, the returned extent will be smaller (zoomed in)
        If ``zoom`` < 0, the returned extent will be larger (zoomed out)

        """
        west_lon, east_lon, south_lat, north_lat = extent

        # Turn number of pluses and minuses into a number than can scale the latitudes and
        # longitudes of our extent
        zoom_multiplier = (1 - 2**-zoom) / 2

        # Calculate bounds for new, zoomed extent
        new_north_lat = north_lat + (south_lat - north_lat) * zoom_multiplier
        new_south_lat = south_lat - (south_lat - north_lat) * zoom_multiplier
        new_east_lon = east_lon + (west_lon - east_lon) * zoom_multiplier
        new_west_lon = west_lon - (west_lon - east_lon) * zoom_multiplier

        return (new_west_lon, new_east_lon, new_south_lat, new_north_lat)

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

            # Set the extent as appropriate based on the area. One special case for 'global'.
            if self.area == 'global':
                self.ax.set_global()
            elif self.area is not None:
                self.ax.set_extent(self.area, ccrs.PlateCarree())

            # Draw all of the plots.
            for p in self.plots:
                with p.hold_trait_notifications():
                    p.draw()

            # Add all of the maps
            if len(self.layers) > len(self.layers_edgecolor):
                self.layers_edgecolor *= len(self.layers)
            if len(self.layers) > len(self.layers_linewidth):
                self.layers_linewidth *= len(self.layers)
            if len(self.layers) > len(self.layers_linestyle):
                self.layers_linestyle *= len(self.layers)
            if len(self.layers) > len(self.layers_zorder):
                self.layers_zorder *= len(self.layers)
            if len(self.layers) > len(self.layers_alpha):
                self.layers_alpha *= len(self.layers)
            for i, feat in enumerate(self._layer_features):
                color = self.layers_edgecolor[i]
                if self.layers[i] in ['', 'land', 'ocean']:
                    color = 'face'
                if self.layers_edgecolor[i] in ['water', 'land', 'land_alt1']:
                    color = get_cartopy_color(self.layers_edgecolor[i])
                width = self.layers_linewidth[i]
                style = self.layers_linestyle[i]
                zorder = self.layers_zorder[i]
                alpha = self.layers_alpha[i]
                kwargs = {'zorder': zorder} if zorder is not None else {}
                self.ax.add_feature(feat, edgecolor=color, linewidth=width,
                                    linestyle=style, alpha=alpha, **kwargs)

            # Use the set title or generate one.
            if (self.right_title is None) and (self.left_title is None):
                title = self.title or ',\n'.join(plot.name for plot in self.plots)
                self.ax.set_title(title, fontsize=self.title_fontsize)
            else:
                if self.title is not None:
                    self.ax.set_title(self.title, fontsize=self.title_fontsize)
                if self.right_title is not None:
                    self.ax.set_title(self.right_title, fontsize=self.title_fontsize,
                                      loc='right')
                if self.left_title is not None:
                    self.ax.set_title(self.left_title, fontsize=self.title_fontsize,
                                      loc='left')
            self._need_redraw = False

    def __copy__(self):
        """Return a copy of this MapPanel."""
        # Create new, blank instance of MapPanel
        cls = self.__class__
        obj = cls.__new__(cls)

        # Copy each attribute from current MapPanel to new MapPanel
        for name in self.trait_names():
            # The 'plots' attribute is a list.
            # A copy must be made for each plot in the list.
            if name == 'plots':
                obj.plots = [copy.copy(plot) for plot in self.plots]
            else:
                setattr(obj, name, getattr(self, name))

        return obj

    def copy(self):
        """Return a copy of the panel."""
        return copy.copy(self)


class SubsetTraits(MetPyHasTraits):
    """Represent common traits for subsetting data."""

    x = Union([Float(allow_none=True, default_value=None), Instance(units.Quantity)])
    x.__doc__ = """The x coordinate of the field to be plotted.

    This is a value with units to choose a desired x coordinate. For example, selecting a
    point or transect through the projection origin, set this parameter to
    ``0 * units.meter``. Note that this requires your data to have an x dimension coordinate.
    """

    longitude = Union([Float(allow_none=True, default_value=None), Instance(units.Quantity)])
    longitude.__doc__ = """The longitude coordinate of the field to be plotted.

    This is a value with units to choose a desired longitude coordinate. For example,
    selecting a point or transect through 95 degrees west, set this parameter to
    ``-95 * units.degrees_east``. Note that this requires your data to have a longitude
    dimension coordinate.
    """

    y = Union([Float(allow_none=True, default_value=None), Instance(units.Quantity)])
    y.__doc__ = """The y coordinate of the field to be plotted.

    This is a value with units to choose a desired x coordinate. For example, selecting a
    point or transect through the projection origin, set this parameter to
    ``0 * units.meter``. Note that this requires your data to have an y dimension coordinate.
    """

    latitude = Union([Float(allow_none=True, default_value=None), Instance(units.Quantity)])
    latitude.__doc__ = """The latitude coordinate of the field to be plotted.

    This is a value with units to choose a desired latitude coordinate. For example,
    selecting a point or transect through 40 degrees north, set this parameter to
    ``40 * units.degrees_north``. Note that this requires your data to have a latitude
    dimension coordinate.
    """

    level = Union([Int(allow_none=True, default_value=None), Instance(units.Quantity)])
    level.__doc__ = """The level of the field to be plotted.

    This is a value with units to choose a desired plot level. For example, selecting the
    850-hPa level, set this parameter to ``850 * units.hPa``. Note that this requires your
    data to have a vertical dimension coordinate.
    """

    time = Instance(datetime, allow_none=True)
    time.__doc__ = """Set the valid time to be plotted as a datetime object.

    If a forecast hour is to be plotted the time should be set to the valid future time, which
    can be done using the `~datetime.datetime` and `~datetime.timedelta` objects
    from the Python standard library. Note that this requires your data to have a time
    dimension coordinate.
    """


@exporter.export
class Plots2D(SubsetTraits):
    """The highest level class related to plotting 2D data.

    This class collects all common methods no matter whether plotting a scalar variable or
    vector. Primary settings common to all types of 2D plots include those for data subsets.
    """

    parent = Instance(Panel)
    _need_redraw = Bool(default_value=True)

    plot_units = Unicode(allow_none=True, default_value=None)
    plot_units.__doc__ = """The desired units to plot the field in.

    Setting this attribute will convert the units of the field variable to the given units for
    plotting using the MetPy Units module.
    """

    scale = Float(default_value=1e0)
    scale.__doc__ = """Scale the field to be plotted by the value given.

    This attribute will scale the field by multiplying by the scale. For example, to
    scale vorticity to be whole values for contouring you could set the scale to 1e5, such that
    the data values will be multiplied by 10^5.
    """

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

        If `image_range` is a matplotlib normalization instance, returns it, otherwise converts
        the tuple image range to a matplotlib normalization instance.

        """
        if isinstance(self.image_range, plt.Normalize):
            return self.image_range
        return plt.Normalize(*self.image_range)

    def clear(self):
        """Clear the plot.

        Resets all internal state and sets need for redraw.

        """
        if getattr(self, 'handle', None) is not None:
            # In matplotlib 3.8, the collections attribute on ContourSet was deprecated.
            # Check for that here so we can avoid the deprecation warning.
            if (not isinstance(ContourSet.__dict__.get('collections'), property)
                    and getattr(self.handle, 'collections', None) is not None):
                self.clear_collections()
            else:
                self.clear_handle()
            self._need_redraw = True

    def clear_handle(self):
        """Clear the handle to the plot instance."""
        self.handle.remove()
        self.handle = None

    def clear_collections(self):
        """Clear the handle collections to the plot instance."""
        for col in self.handle.collections:
            col.remove()
        self.handle = None

    @observe('parent')
    def _parent_changed(self, _):
        """Handle setting the parent object for the plot."""
        self.clear()

    @observe('x', 'longitude', 'y', 'latitude', 'level', 'time')
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
        """Xarray dataset that contains the field to be plotted."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        self._update_data()

    @property
    def name(self):
        """Generate a name for the plot."""
        if isinstance(self.field, tuple):
            ret = ''
            ret += ' and '.join(self.field)
        else:
            ret = self.field
        if self.level is not None:
            ret += f'@{self.level:d}'
        return ret

    def copy(self):
        """Return a copy of the plot."""
        return copy.copy(self)


@exporter.export
class PlotScalar(Plots2D):
    """Defines the common elements of 2D scalar plots for single scalar value fields.

    Most of the other traits here are for one or more of the specific plots. Currently this
    allows too many options for `ContourPlot` since it does not user image_range, for
    example. Similar issues for `ImagePlot` and `FilledContourPlot`.
    """

    field = Unicode()
    field.__doc__ = """Name of the field to be plotted.

    This is the name of the variable from the dataset that is to be plotted. An example,
    from a model grid file that uses the THREDDS convention for naming would be
    `Geopotential_height_isobaric` or `Temperature_isobaric`. For GOES-16/17 satellite data it
    might be `Sectorized_CMI`. To check for the variables available within a dataset, list the
    variables with the following command assuming the dataset was read using xarray as `ds`,
    `list(ds)`
    """

    smooth_field = Int(allow_none=True, default_value=None)
    smooth_field.__doc__ = """Number of smoothing passes using 9-pt smoother.

    By setting this parameter with an integer value it will call the MetPy 9-pt smoother and
    provide a smoothed field for plotting. It is best to use this smoothing for data with
    finer resolutions (e.g., smaller grid spacings with a lot of grid points).

    See Also
    --------
    metpy.calc.smooth_n_point, smooth_contour
    """

    smooth_contour = Union([Int(allow_none=True, default_value=None),
                            Tuple(Int(allow_none=True, default_value=None),
                                  Int(allow_none=True, default_value=None))])
    smooth_contour.__doc__ = """Spline interpolation to smooth contours.

    This attribute requires settings for the `metpy.calc.zoom_xarray` function, which will
    produce a spline interpolation given an integer zoom factor. Either a single integer
    specifying the zoom factor (e.g., 4) or a tuple containing two integers for the zoom factor
    and the spline interpolation order can be used. The default spline interpolation order is
    3.

    This is best used to smooth contours when contouring a sparse grid (e.g., when your data
    has a large grid spacing).

    See Also
    --------
    metpy.calc.zoom_xarray, smooth_field
    """

    @observe('field')
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._griddata = None
        self.clear()

    @property
    def griddata(self):
        """Return the internal cached data."""
        if getattr(self, '_griddata', None) is None:

            # Select our particular field of interest
            if self.field:
                data = self.data.metpy.parse_cf(self.field)
            elif hasattr(self.data.metpy, 'parse_cf'):
                # Handles the case where we have a dataset but no specified field
                raise ValueError('field attribute has not been set.')
            else:
                data = self.data

            # Subset to 2D using MetPy's fancy .sel
            subset = {'method': 'nearest'}
            for dim_coord in ('x', 'longitude', 'y', 'latitude', 'vertical', 'time'):
                selector = self.level if dim_coord == 'vertical' else getattr(self, dim_coord)
                if selector is not None:
                    subset[dim_coord] = selector
            data_subset = data.metpy.sel(**subset).squeeze()
            if (data_subset.ndim != 2):
                if data_subset.ndim == 3:
                    if (data_subset.shape[-1] not in (3, 4)):
                        raise ValueError(
                            'Must provide a combination of subsetting values to give either 2D'
                            ' data or 3D data subset for plotting with third dimension size 3'
                            ' or 4'
                        )
                else:
                    raise ValueError(
                        'Must provide a combination of subsetting values to give 2D data '
                        'subset for plotting'
                    )
            # Handle unit conversion (both direct unit specification and scaling)
            if self.plot_units is not None:
                data_subset = data_subset.metpy.convert_units(self.plot_units)

            # Handle smoothing of data
            if self.smooth_field is not None:
                data_subset = smooth_n_point(data_subset, 9, self.smooth_field)
            # Handle zoom interpolation
            if self.smooth_contour is not None:
                if isinstance(self.smooth_contour, tuple):
                    zoom = self.smooth_contour[0]
                    order = self.smooth_contour[1]
                else:
                    zoom = self.smooth_contour
                    order = 3
                data_subset = zoom_xarray(data_subset, zoom, order=order)

            self._griddata = data_subset * self.scale

        return self._griddata

    @property
    def plotdata(self):
        """Return the data for plotting.

        The two dimension coordinates and the data array.

        """
        try:
            plot_x_dim = self.griddata.metpy.find_axis_number('x')
            plot_y_dim = self.griddata.metpy.find_axis_number('y')
        except ValueError:
            plot_x_dim = 1
            plot_y_dim = 0

        return (
            self.griddata[self.griddata.dims[plot_x_dim]],
            self.griddata[self.griddata.dims[plot_y_dim]],
            self.griddata
        )

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, 'handle', None) is None:
                self._build()
            if getattr(self, 'colorbar', None) is not None:
                if isinstance(self.colorbar, dict):
                    cbar = self.parent.ax.figure.colorbar(
                        self.handle, **self.colorbar)
                else:
                    cbar = self.parent.ax.figure.colorbar(
                        self.handle, orientation=self.colorbar, pad=0, aspect=50)
                cbar.ax.tick_params(labelsize=self.colorbar_fontsize)
            self._need_redraw = False


class ContourTraits(MetPyHasTraits):
    """Represents common contour traits."""

    contours = Union([List(Float()), Int(), Instance(range)], default_value=25)
    contours.__doc__ = """A list of values to contour or an integer number of contour levels.

    This parameter sets contour or colorfill values for a plot. Values can be entered either
    as a Python range instance, a list of values or as an integer with the number of contours
    to be plotted (as per matplotlib documentation). A list can be generated by using square
    brackets or creating a numpy 1D array and converting it to a list with the
    `~numpy.ndarray.tolist` method.
    """

    clabels = Bool(default_value=False)
    clabels.__doc__ = """A boolean (True/False) on whether to plot contour labels.

    To plot contour labels set this trait to ``True``, the default value is ``False``.
    """

    label_fontsize = Union([Int(), Float(), Unicode()], allow_none=True, default_value=None)
    label_fontsize.__doc__ = """An integer, float, or string value to set the font size of
    labels for contours.

    This trait sets the font size for labels that will plot along contour lines. Accepts
    size in points or relative size. Allowed relative sizes are those of Matplotlib:
    'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    """


class ColorfillTraits(MetPyHasTraits):
    """Represent common colorfill traits."""

    colormap = Unicode(allow_none=True, default_value=None)
    colormap.__doc__ = """The string name for a Matplolib or MetPy colormap.

    For example, the Blue-Purple colormap from Matplotlib can be accessed using 'BuPu'.
    """

    image_range = Union([Tuple(Float(allow_none=True), Float(allow_none=True)),
                         Instance(plt.Normalize)], default_value=(None, None))
    image_range.__doc__ = """A tuple of min and max values that represent the range of values
    to color the rasterized image.

    The min and max values entered as a tuple will be converted to a
    `matplotlib.colors.Normalize` instance for plotting.
    """

    colorbar = Union([Unicode(default_value=None, allow_none=True), Dict()])
    colorbar.__doc__ = """A string (horizontal/vertical) on whether to add a colorbar to the
    plot.

    To add a colorbar associated with the plot, you can either set the trait with a string of
    ``horizontal`` or ``vertical``, which specifies the orientation of the produced colorbar
    and uses pre-defined defaults for aspect and pad. Alternatively, you can set a dictionary
    of keyword argument values valid for a Matplotlib colorbar to specify how the colorbar will
    be plotted. The default value is ``None``.
    """

    colorbar_fontsize = Union([Int(), Float(), Unicode()], allow_none=True, default_value=None)
    colorbar_fontsize.__doc__ = """An integer, float, or string value to set the font size of
    labels for the colorbar.

    This trait sets the font size of labels for the colorbar. Accepts size in points or
    relative size. Allowed relative sizes are those of Matplotlib: 'xx-small', 'x-small',
    'small', 'medium', 'large', 'x-large', 'xx-large'.
    """


@exporter.export
class ImagePlot(PlotScalar, ColorfillTraits, ValidationMixin):
    """Make raster image using `~matplotlib.pyplot.imshow` for satellite or colored image."""

    @observe('colormap', 'image_range')
    def _set_need_redraw(self, _):
        """Handle changes to attributes that just need a simple redraw."""
        if hasattr(self, 'handle'):
            self.handle.set_cmap(self._cmap_obj)
            self.handle.set_norm(self._norm_obj)
            self._need_redraw = True

    @observe('colorbar')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    @property
    def plotdata(self):
        """Return the data for plotting.

        The two dimension coordinates and the data array

        """
        x_like = self.griddata[self.griddata.dims[1]]

        # At least currently imshow with cartopy does not like this
        if 'degree' in x_like.units:
            x_like = x_like.data
            x_like[x_like > 180] -= 360

        return x_like, self.griddata[self.griddata.dims[0]], self.griddata

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x_like, y_like, imdata = self.plotdata

        kwargs = plot_kwargs(imdata, self.mpl_args)

        # If we're on a map, we use min/max for y and manually figure out origin to try to
        # avoid upside down images created by images where y[0] > y[-1], as well as
        # specifying the transform
        # TODO item() shouldn't be necessary, but is until pydata/xarray#9043 is fixed.
        kwargs['extent'] = (x_like[0].item(), x_like[-1].item(),
                            y_like.min().item(), y_like.max().item())
        kwargs['origin'] = 'upper' if y_like[0] > y_like[-1] else 'lower'
        kwargs.setdefault('cmap', self._cmap_obj)
        kwargs.setdefault('norm', self._norm_obj)

        self.handle = self.parent.ax.imshow(imdata, **kwargs)


@exporter.export
class ContourPlot(PlotScalar, ContourTraits, ValidationMixin):
    """Make contour plots by defining specific traits."""

    linecolor = Unicode('black')
    linecolor.__doc__ = """A string value to set the color of plotted contours; default is
    black.

    This trait can be set to any Matplotlib color
    (https://matplotlib.org/3.1.0/gallery/color/named_colors.html)
    """

    linewidth = Int(2)
    linewidth.__doc__ = """An integer value to set the width of plotted contours; default value
    is 2.

    This trait changes the thickness of contour lines with a higher value plotting a thicker
    line.
    """

    linestyle = Unicode(None, allow_none=True)
    linestyle.__doc__ = """A string value to set the linestyle (e.g., dashed), or `None`;
    default is `None`, which, when using monochrome line colors, uses solid lines for positive
    values and dashed lines for negative values.

    The valid string values are those of Matplotlib which are 'solid', 'dashed', 'dotted', and
    'dashdot', as well as their short codes ('-', '--', '.', '-.'). The object `None`, as
    described above, can also be used.
    """

    @observe('contours', 'linecolor', 'linewidth', 'linestyle', 'clabels', 'label_fontsize')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x_like, y_like, imdata = self.plotdata

        kwargs = plot_kwargs(imdata, self.mpl_args)
        kwargs.setdefault('linewidths', self.linewidth)
        kwargs.setdefault('colors', self.linecolor)
        kwargs.setdefault('linestyles', self.linestyle)

        self.handle = self.parent.ax.contour(x_like, y_like, imdata.metpy.dequantify(),
                                             self.contours, **kwargs)
        if self.clabels:
            self.handle.clabel(inline=1, fmt='%.0f', inline_spacing=8,
                               use_clabeltext=True, fontsize=self.label_fontsize)


@exporter.export
class FilledContourPlot(PlotScalar, ColorfillTraits, ContourTraits, ValidationMixin):
    """Make color-filled contours plots by defining appropriate traits."""

    @observe('contours', 'colorbar', 'colormap')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling any plotting methods as necessary."""
        x_like, y_like, imdata = self.plotdata

        kwargs = plot_kwargs(imdata, self.mpl_args)
        kwargs.setdefault('cmap', self._cmap_obj)
        kwargs.setdefault('norm', self._norm_obj)

        self.handle = self.parent.ax.contourf(x_like, y_like, imdata, self.contours, **kwargs)


@exporter.export
class RasterPlot(PlotScalar, ColorfillTraits):
    """Make raster plots by defining relevant traits."""

    @observe('image_range', 'colorbar', 'colormap')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of pcolormesh()
        self.clear()

    def _build(self):
        """Build the raster plot by calling any plotting methods as necessary."""
        x_like, y_like, imdata = self.plotdata

        kwargs = plot_kwargs(imdata, self.mpl_args)
        kwargs.setdefault('cmap', self._cmap_obj)
        kwargs.setdefault('norm', self._norm_obj)

        self.handle = self.parent.ax.pcolormesh(x_like, y_like, imdata, **kwargs)


@exporter.export
class PlotVector(Plots2D):
    """Defines common elements for 2D vector plots.

    This class collects common elements including the field trait, which is a tuple argument
    accepting two strings, for plotting 2D vector fields.
    """

    field = Tuple(Unicode(), Unicode())
    field.__doc__ = """A tuple containing the two components of the vector field from the
    dataset in the form (east-west component, north-south component).

    For a wind barb plot each component of the wind must be specified and should be of the form
    (u-wind, v-wind).
    """

    pivot = Unicode('middle')
    pivot.__doc__ = """A string setting the pivot point of the vector. Default value is
    'middle'.

    This trait takes the values of the keyword argument from `matplotlin.pyplot.barbs`:
    'tip' or 'middle'.
    """

    skip = Tuple(Int(), Int(), default_value=(1, 1))
    skip.__doc__ = """A tuple of integers to indicate the number of grid points to skip between
    plotting vectors. Default is (1, 1).

    This trait is to be used to reduce the number of vectors plotted in the (east-west,
    north-south) components. The two values can be set to the same or different integer values
    depending on what is desired.
    """

    earth_relative = Bool(default_value=True)
    earth_relative.__doc__ = """A boolean value to indicate whether the vector to be plotted
    is earth- or grid-relative. Default value is `True`, indicating that vectors are
    earth-relative.

    Common gridded meteorological datasets including GFS and NARR output contain wind
    components that are earth-relative. The primary exception is NAM output with wind
    components that are grid-relative. For any grid-relative vectors set this trait to
    `False`. This value is ignored for 2D vector fields not in the plane of the plot (e.g.,
    cross sections).
    """

    color = Unicode(default_value='black')
    color.__doc__ = """A string value that controls the color of the vectors. Default value is
    black.

    This trait can be set to any named color from
    `Matplotlibs Colors <https://matplotlib.org/3.1.0/gallery/color/named_colors.html>`
    """

    @observe('field')
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._griddata_u = None
        self._griddata_v = None
        self.clear()

    @property
    def griddata(self):
        """Return the internal cached data."""
        if getattr(self, '_griddata_u', None) is None:

            if not self.field[0]:
                raise ValueError('field attribute not set correctly')

            u = self.data.metpy.parse_cf(self.field[0])
            v = self.data.metpy.parse_cf(self.field[1])

            # Subset to 2D using MetPy's fancy .sel
            subset = {'method': 'nearest'}
            for dim_coord in ('x', 'longitude', 'y', 'latitude', 'vertical', 'time'):
                selector = self.level if dim_coord == 'vertical' else getattr(self, dim_coord)
                if selector is not None:
                    subset[dim_coord] = selector
            data_subset_u = u.metpy.sel(**subset).squeeze()
            data_subset_v = v.metpy.sel(**subset).squeeze()
            if data_subset_u.ndim != 2 or data_subset_v.ndim != 2:
                raise ValueError(
                    'Must provide a combination of subsetting values to give 2D data subsets '
                    'for plotting'
                )

            if self.plot_units is not None:
                data_subset_u = data_subset_u.metpy.convert_units(self.plot_units)
                data_subset_u = data_subset_u.metpy.dequantify()
                data_subset_v = data_subset_v.metpy.convert_units(self.plot_units)
                data_subset_v = data_subset_v.metpy.dequantify()
            self._griddata_u = data_subset_u * self.scale
            self._griddata_v = data_subset_v * self.scale

        return self._griddata_u, self._griddata_v

    @property
    def plotdata(self):
        """Return the data for plotting.

        The dimension coordinates and data arrays.

        """
        check_earth_relative = False
        try:
            plot_x_dim = self.griddata[0].metpy.find_axis_number('x')
            plot_y_dim = self.griddata[0].metpy.find_axis_number('y')
            check_earth_relative = True
        except ValueError:
            plot_x_dim = 1
            plot_y_dim = 0

        x_like = self.griddata[0][self.griddata[0].dims[plot_x_dim]]
        y_like = self.griddata[0][self.griddata[0].dims[plot_y_dim]]

        if check_earth_relative:
            # Conditionally apply earth v. grid relative adjustments if we are in the plane of
            # the plot
            # TODO: this seems like it could use a refactor to be more explicit about what
            # coords are grid x and y vs latitude and longitude (both for code readability and
            # error-proneness).
            x, y = x_like, y_like
            if self.earth_relative:
                x, y, _ = ccrs.PlateCarree().transform_points(
                    self.griddata[0].metpy.cartopy_crs,
                    *np.meshgrid(x, y)
                ).T
                x_like = x.T
                y_like = y.T
            else:
                if 'degree' in x.units:
                    x, y, _ = self.griddata[0].metpy.cartopy_crs.transform_points(
                        ccrs.PlateCarree(), *np.meshgrid(x, y)).T
                    x_like = x.T
                    y_like = y.T

        if x_like.ndim == 1:
            x_like, y_like = np.meshgrid(x_like, y_like)

        return x_like, y_like, self.griddata[0], self.griddata[1]

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, 'handle', None) is None:
                self._build()
            self._need_redraw = False


@exporter.export
class BarbPlot(PlotVector, ValidationMixin):
    """Make plots of wind barbs on a map with traits to refine the look of plotted elements."""

    barblength = Float(default_value=7)
    barblength.__doc__ = """A float value that changes the length of the wind barbs. Default
    value is 7.

    This trait corresponds to the keyword length in `matplotlib.pyplot.barbs`.
    """

    @observe('barblength', 'pivot', 'skip', 'earth_relative', 'color')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling needed plotting methods as necessary."""
        x_like, y_like, u, v = self.plotdata

        kwargs = plot_kwargs(u, self.mpl_args)
        kwargs.setdefault('color', self.color)
        kwargs.setdefault('pivot', self.pivot)
        kwargs.setdefault('length', self.barblength)
        kwargs.setdefault('zorder', 2)

        # Conditionally apply the proper transform
        if 'transform' in kwargs and self.earth_relative:
            kwargs['transform'] = ccrs.PlateCarree()

        wind_slice = (slice(None, None, self.skip[0]), slice(None, None, self.skip[1]))

        self.handle = self.parent.ax.barbs(
            x_like[wind_slice], y_like[wind_slice],
            u.values[wind_slice], v.values[wind_slice],
            **kwargs)


@exporter.export
class ArrowPlot(PlotVector, ValidationMixin):
    """Make plots of wind barbs on a map with traits to refine the look of plotted elements."""

    arrowscale = Union([Int(), Float(), Unicode()], allow_none=True, default_value=None)
    arrowscale.__doc__ = """Number of data units per arrow length unit, e.g., m/s per plot
    width; a smaller scale parameter makes the arrow longer. Default is `None`.

    If `None`, a simple autoscaling algorithm is used, based on the average
    vector length and the number of vectors. The arrow length unit is given by
    the `key_length` attribute.

    This trait corresponds to the keyword length in `matplotlib.pyplot.quiver`.
    """

    arrowkey = Tuple(Float(allow_none=True), Float(allow_none=True), Float(allow_none=True),
                     Unicode(allow_none=True), Unicode(allow_none=True), default_value=None,
                     allow_none=True)
    arrowkey.__doc__ = """Set the characteristics of an arrow key using a tuple of values
    representing (value, xloc, yloc, position, string).

    Default is `None`.

    If `None`, no vector key will be plotted.

    value default is 100
    xloc default is 0.85
    yloc default is 1.02
    position default is 'E' (options are 'N', 'S', 'E', 'W')
    label default is an empty string

    If you wish to change a characteristic of the arrowkey you'll need to have a tuple of five
    elements, fill in the full tuple using `None` for those characteristics you wish to use the
    default value and put in the new values for the other elements. This trait corresponds to
    the keyword length in `matplotlib.pyplot.quiverkey`.
    """

    @observe('arrowscale', 'pivot', 'skip', 'earth_relative', 'color', 'arrowkey')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of quiver()
        self.clear()

    def _build(self):
        """Build the plot by calling needed plotting methods as necessary."""
        x_like, y_like, u, v = self.plotdata

        kwargs = plot_kwargs(u, self.mpl_args)
        kwargs.setdefault('color', self.color)
        kwargs.setdefault('pivot', self.pivot)
        kwargs.setdefault('scale', self.arrowscale)

        # Conditionally apply the proper transform
        if 'transform' in kwargs and self.earth_relative:
            kwargs['transform'] = ccrs.PlateCarree()

        wind_slice = (slice(None, None, self.skip[0]), slice(None, None, self.skip[1]))

        self.handle = self.parent.ax.quiver(
            x_like[wind_slice], y_like[wind_slice],
            u.values[wind_slice], v.values[wind_slice],
            **kwargs)

        # The order here needs to match the order of the tuple
        if self.arrowkey is not None:
            key_kwargs = {'U': 100, 'X': 0.85, 'Y': 1.02, 'labelpos': 'E', 'label': ''}
            for name, val in zip(key_kwargs, self.arrowkey):
                if val is not None:
                    key_kwargs[name] = val
            self.parent.ax.quiverkey(self.handle, labelcolor=self.color, **key_kwargs)


@exporter.export
class PlotObs(MetPyHasTraits, ValidationMixin):
    """The highest level class related to plotting observed surface and upperair data.

    This class collects all common methods no matter whether plotting a upper-level or
    surface data using station plots.

    List of Traits:
      * level
      * time
      * fields
      * locations (optional)
      * time_window (optional)
      * formats (optional)
      * colors (optional)
      * plot_units (optional)
      * vector_field (optional)
      * vector_field_color (optional)
      * vector_field_length (optional)
      * vector_plot_units (optional)
      * reduce_points (optional)
      * fontsize (optional)
    """

    parent = Instance(Panel)
    _need_redraw = Bool(default_value=True)

    level = Union([Int(allow_none=True), Instance(units.Quantity)], default_value=None)
    level.__doc__ = """The level of the field to be plotted.

    This is a value with units to choose the desired plot level. For example, selecting the
    850-hPa level, set this parameter to ``850 * units.hPa``. For surface data, parameter
    must be set to `None`.
    """

    time = Instance(datetime, allow_none=True)
    time.__doc__ = """Set the valid time to be plotted as a datetime object.

    If a forecast hour is to be plotted the time should be set to the valid future time, which
    can be done using the `~datetime.datetime` and `~datetime.timedelta` objects
    from the Python standard library.
    """

    time_window = Instance(timedelta, default_value=timedelta(minutes=0), allow_none=True)
    time_window.__doc__ = """Set a range to look for data to plot as a timedelta object.

    If this parameter is set, it will subset the data provided to be within the time and plus
    or minus the range value given. If there is more than one observation from a given station
    then it will keep only the most recent one for plotting purposes. Default value is to have
    no range. (optional)
    """

    fields = List(Unicode())
    fields.__doc__ = """Name of the scalar or symbol fields to be plotted.

    List of parameters to be plotted around station plot (e.g., temperature, dewpoint, skyc).
    """

    locations = List(default_value=['C'])
    locations.__doc__ = """List of strings for scalar or symbol field plotting locations.

    List of parameters locations for plotting parameters around the station plot (e.g.,
    NW, NE, SW, SE, W, C). (optional)
    """

    formats = List(default_value=[None])
    formats.__doc__ = """List of the scalar, symbol, and text field data formats. (optional)

    List of scalar parameters formatters or mapping values (if symbol) for plotting text and/or
    symbols around the station plot (e.g., for pressure variable
    ```lambda v: format(10 * v, '.0f')[-3:]```).

    For symbol mapping the following options are available to be put in as a string:
    current_weather, sky_cover, low_clouds, mid_clouds, high_clouds, and pressure_tendency.

    For plotting text, use the format setting of 'text'.
    """

    colors = List(Unicode(), default_value=['black'])
    colors.__doc__ = """List of the scalar and symbol field colors.

    List of strings that represent the colors to be used for the variable being plotted.
    (optional)
    """

    vector_field = List(default_value=[None], allow_none=True)
    vector_field.__doc__ = """List of the vector field to be plotted.

    List of vector components to combined and plotted from the center of the station plot
    (e.g., wind components). (optional)
    """

    vector_field_color = Unicode('black', allow_none=True)
    vector_field_color.__doc__ = """String color name to plot the vector. (optional)"""

    vector_field_length = Int(default_value=None, allow_none=True)
    vector_field_length.__doc__ = """Integer value to set the length of the plotted vector.
    (optional)
    """

    reduce_points = Float(default_value=0)
    reduce_points.__doc__ = """Float to reduce number of points plotted. (optional)"""

    plot_units = List(default_value=[None], allow_none=True)
    plot_units.__doc__ = """A list of the desired units to plot the fields in.

    Setting this attribute will convert the units of the field variable to the given units for
    plotting using the MetPy Units module, provided that units are attached to the DataFrame.
    """

    vector_plot_units = Unicode(default_value=None, allow_none=True)
    vector_plot_units.__doc__ = """The desired units to plot the vector field in.

    Setting this attribute will convert the units of the field variable to the given units for
    plotting using the MetPy Units module, provided that units are attached to the DataFrame.
    """

    fontsize = Int(10)
    fontsize.__doc__ = """An integer value to set the font size of station plots. Default
    is 10 pt."""

    def clear(self):
        """Clear the plot.

        Resets all internal state and sets need for redraw.

        """
        if getattr(self, 'handle', None) is not None:
            self.handle.ax.cla()
            self.handle = None
            self._need_redraw = True

    @observe('parent')
    def _parent_changed(self, _):
        """Handle setting the parent object for the plot."""
        self.clear()

    @observe('fields', 'level', 'time', 'vector_field', 'time_window')
    def _update_data(self, _=None):
        """Handle updating the internal cache of data.

        Responds to changes in various subsetting parameters.

        """
        self._obsdata = None
        self.clear()

    # Can't be a Traitlet because notifications don't work with arrays for traits
    # notification never happens
    @property
    def data(self):
        """Pandas dataframe that contains the fields to be plotted."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        self._update_data()

    @property
    def name(self):
        """Generate a name for the plot."""
        ret = ''
        ret += ' and '.join(self.fields)
        if self.level is not None:
            ret += f'@{self.level:d}'
        return ret

    @property
    def obsdata(self):
        """Return the internal cached data."""
        if getattr(self, '_obsdata', None) is None:
            # Use a copy of data so we retain all of the original data passed in unmodified
            data = self.data

            # Subset for a particular level if given
            if self.level is not None:
                mag = getattr(self.level, 'magnitude', self.level)
                data = data[data.pressure == mag]

            # Subset for our particular time
            if self.time is not None:
                # If data are not currently indexed by time, we need to do so choosing one of
                # the columns we're looking for
                if not isinstance(data.index, pd.DatetimeIndex):
                    time_vars = ['valid', 'time', 'valid_time', 'date_time', 'date']
                    dim_times = [time_var for time_var in time_vars if
                                 time_var in list(self.data)]
                    if not dim_times:
                        raise AttributeError(
                            'Time variable not found. Valid variable names are:'
                            f'{time_vars}')

                    data = data.set_index(dim_times[0])
                    if not isinstance(data.index, pd.DatetimeIndex):
                        # Convert our column of interest to a datetime
                        data = data.reset_index()
                        time_index = pd.to_datetime(data[dim_times[0]])
                        data = data.set_index(time_index)

                # Works around the fact that traitlets 4.3 insists on sending us None by
                # default because timedelta(0) is Falsey.
                window = timedelta(minutes=0) if self.time_window is None else self.time_window

                # Indexes need to be properly sorted for the slicing below to work; the
                # error you get if that's not the case really convoluted, which is why
                # we don't rely on users doing it.
                data = data.sort_index()
                data = data[self.time - window:self.time + window]

            # Look for the station column
            stn_vars = ['station', 'stn', 'station_id', 'stid']
            dim_stns = [stn_var for stn_var in stn_vars if stn_var in list(self.data)]
            if not dim_stns:
                raise AttributeError('Station variable not found. Valid variable names are: '
                                     f'{stn_vars}')
            else:
                dim_stn = dim_stns[0]

            # Make sure we only use one observation per station
            self._obsdata = data.groupby(dim_stn).tail(1)

        return self._obsdata

    @property
    def plotdata(self):
        """Return the data for plotting.

        The data arrays, x coordinates, and y coordinates.

        """
        plot_data = {}
        for dim_name in list(self.obsdata):
            if dim_name.find('lat') != -1:
                lat = self.obsdata[dim_name]
            elif dim_name.find('lon') != -1:
                lon = self.obsdata[dim_name]
            else:
                plot_data[dim_name] = self.obsdata[dim_name]
        return lon.values, lat.values, plot_data

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, 'handle', None) is None:
                self._build()
            self._need_redraw = False

    @observe('colors', 'formats', 'locations', 'reduce_points', 'vector_field_color')
    def _set_need_rebuild(self, _):
        """Handle changes to attributes that need to regenerate everything."""
        # Because matplotlib doesn't let you just change these properties, we need
        # to trigger a clear and re-call of contour()
        self.clear()

    def _build(self):
        """Build the plot by calling needed plotting methods as necessary."""
        lon, lat, data = self.plotdata

        # Use the cartopy map projection to transform station locations to the map and
        # then refine the number of stations plotted by setting a radius
        scale = 1. if self.parent._proj_obj == ccrs.PlateCarree() else 100000.
        point_locs = self.parent._proj_obj.transform_points(ccrs.PlateCarree(), lon, lat)
        subset = reduce_point_density(point_locs, self.reduce_points * scale)
        kwargs = self.mpl_args
        kwargs.setdefault('clip_on', True)
        kwargs.setdefault('transform', ccrs.PlateCarree())
        kwargs.setdefault('fontsize', self.fontsize)

        self.handle = StationPlot(self.parent.ax, lon[subset], lat[subset], **kwargs)

        for i, ob_type in enumerate(self.fields):
            field_kwargs = {}
            location = self.locations[i] if len(self.locations) > 1 else self.locations[0]
            if len(self.colors) > 1:
                field_kwargs['color'] = self.colors[i]
            else:
                field_kwargs['color'] = self.colors[0]
            if len(self.formats) > 1:
                field_kwargs['formatter'] = self.formats[i]
            else:
                field_kwargs['formatter'] = self.formats[0]
            if len(self.plot_units) > 1:
                field_kwargs['plot_units'] = self.plot_units[i]
            else:
                field_kwargs['plot_units'] = self.plot_units[0]
            if hasattr(self.data, 'units') and (field_kwargs['plot_units'] is not None):
                parameter = units.Quantity(data[ob_type][subset].values,
                                           self.data.units[ob_type])
            else:
                parameter = data[ob_type][subset]
            if field_kwargs['formatter'] is not None:
                mapper = getattr(wx_symbols, str(field_kwargs['formatter']), None)
                if mapper is not None:
                    field_kwargs.pop('formatter')
                    self.handle.plot_symbol(location, parameter, mapper, **field_kwargs)
                else:
                    if self.formats[i] == 'text':
                        self.handle.plot_text(location, parameter, color=field_kwargs['color'])
                    else:
                        self.handle.plot_parameter(location, parameter, **field_kwargs)
            else:
                field_kwargs.pop('formatter')
                self.handle.plot_parameter(location, parameter, **field_kwargs)

        if self.vector_field[0] is not None:
            vector_kwargs = {
                'color': self.vector_field_color,
                'plot_units': self.vector_plot_units,
            }

            if hasattr(self.data, 'units') and (vector_kwargs['plot_units'] is not None):
                u = units.Quantity(data[self.vector_field[0]][subset].values,
                                   self.data.units[self.vector_field[0]])
                v = units.Quantity(data[self.vector_field[1]][subset].values,
                                   self.data.units[self.vector_field[1]])
            else:
                vector_kwargs.pop('plot_units')
                u = data[self.vector_field[0]][subset]
                v = data[self.vector_field[1]][subset]
            if self.vector_field_length is not None:
                vector_kwargs['length'] = self.vector_field_length
            self.handle.plot_barb(u, v, **vector_kwargs)

    def copy(self):
        """Return a copy of the plot."""
        return copy.copy(self)


@exporter.export
class PlotGeometry(MetPyHasTraits):
    """Plot collections of Shapely objects and customize their appearance."""

    parent = Instance(Panel)
    _need_redraw = Bool(default_value=True)

    geometry = Instance(collections.abc.Iterable, allow_none=False)
    geometry.__doc__ = """A collection of Shapely objects to plot.

    A collection of Shapely objects, such as the 'geometry' column from a
    ``geopandas.GeoDataFrame``. Acceptable Shapely objects are ``shapely.MultiPolygon``,
    ``shapely.Polygon``, ``shapely.MultiLineString``, ``shapely.LineString``,
    ``shapely.MultiPoint``, and ``shapely.Point``.
    """

    fill = Union([Instance(collections.abc.Iterable), Unicode()], default_value=['lightgray'],
                 allow_none=True)
    fill.__doc__ = """Fill color(s) for polygons and points.

    A single string (color name or hex code) or collection of strings with which to fill
    polygons and points. If a collection, the first color corresponds to the first Shapely
    object in `geometry`, the second color corresponds to the second Shapely object, and so on.
    If `fill` is shorter than `geometry`, `fill` cycles back to the beginning, repeating the
    sequence of colors as needed. Default value is lightgray.
    """

    stroke = Union([Instance(collections.abc.Iterable), Unicode()], default_value=['black'],
                   allow_none=True)
    stroke.__doc__ = """Stroke color(s) for polygons and line color(s) for lines.

    A single string (color name or hex code) or collection of strings with which to outline
    polygons and color lines. If a collection, the first color corresponds to the first Shapely
    object in `geometry`, the second color corresponds to the second Shapely object, and so on.
    If `stroke` is shorter than `geometry`, `stroke` cycles back to the beginning, repeating
    the sequence of colors as needed. Default value is black.
    """

    stroke_width = Union([Instance(collections.abc.Iterable), Float()], default_value=[1],
                         allow_none=True)
    stroke_width.__doc__ = """Stroke width(s) for polygons and lines.

    A single integer or floating point value or collection of values representing the size of
    the stroke width. If a collection, the first value corresponds to the first Shapely
    object in `geometry`, the second value corresponds to the second Shapely object, and so on.
    If `stroke_width` is shorter than `geometry`, `stroke_width` cycles back to the beginning,
    repeating the sequence of values as needed. Default value is 1.
    """

    marker = Unicode(default_value='.', allow_none=False)
    marker.__doc__ = """Symbol used to denote points.

    Accepts any matplotlib marker. Default value is '.', which plots a dot at each point.
    """

    labels = Instance(collections.abc.Iterable, allow_none=True)
    labels.__doc__ = """A collection of labels corresponding to plotted geometry.

    A collection of strings to use as labels for geometry, such as a column from a
    ``Geopandas.GeoDataFrame``. The first label corresponds to the first Shapely object in
    `geometry`, the second label corresponds to the second Shapely object, and so on. The
    length of `labels` must be equal to the length of `geometry`. Labels are positioned along
    the edge of polygons, and below lines and points. No labels are plotted if this attribute
    is left undefined, or set equal to `None`.
    """

    label_fontsize = Union([Int(), Float(), Unicode()], default_value=None, allow_none=True)
    label_fontsize.__doc__ = """An integer or string value for the font size of labels.

    Accepts size in points or relative size. Allowed relative sizes are those of Matplotlib:
    'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'.
    """

    label_facecolor = Union([Instance(collections.abc.Iterable), Unicode()], allow_none=True)
    label_facecolor.__doc__ = """Font color(s) for labels.

    A single string (color name or hex code) or collection of strings for the font color of
    labels. If a collection, the first color corresponds to the label of the first Shapely
    object in `geometry`, the second color corresponds to the label of the second Shapely
    object, and so on. Default value is `stroke`.
    """

    label_edgecolor = Union([Instance(collections.abc.Iterable), Unicode()], allow_none=True)
    label_edgecolor.__doc__ = """Outline color(s) for labels.

    A single string (color name or hex code) or collection of strings for the outline color of
    labels. If a collection, the first color corresponds to the label of the first Shapely
    object in `geometry`, the second color corresponds to the label of the second Shapely
    object, and so on. Default value is `fill`.
    """

    @staticmethod
    @validate('geometry')
    def _valid_geometry(_, proposal):
        """Cast `geometry` into a list once it is provided by user.

        Users can provide any kind of collection, such as a ``GeoPandas.GeoSeries``, and this
        turns them into a list.
        """
        geometry = proposal['value']
        return list(geometry)

    @staticmethod
    @validate('fill', 'stroke', 'label_facecolor', 'label_edgecolor')
    def _valid_color_list(_, proposal):
        """Cast color-related attributes into a list once provided by user.

        This is necessary because _build() expects to cycle through a list of colors when
        assigning them to the geometry.
        """
        color = proposal['value']

        # `color` must be a collection if it is not a string
        return [color] if isinstance(color, str) else list(color)

    @staticmethod
    @validate('labels')
    def _valid_labels(_, proposal):
        """Cast `labels` into a list once provided by user."""
        labels = proposal['value']
        return list(labels)

    @observe('fill', 'stroke')
    def _update_label_colors(self, change):
        """Set default text colors using `fill` and `stroke`.

        If `label_facecolor` or `label_edgecolor` have not been specified, provide default
        colors for those attributes using `fill` and `stroke`.
        """
        if change['name'] == 'fill' and self.label_edgecolor is None:
            self.label_edgecolor = self.fill
        elif change['name'] == 'stroke' and self.label_facecolor is None:
            self.label_facecolor = self.stroke

    @property
    def name(self):
        """Generate a name for the plot."""
        # Unlike Plots2D and PlotObs, there are no other attributes (such as 'fields' or
        # 'levels') from which to name the plot. A generic name is returned here in case the
        # user does not provide their own title, in which case MapPanel.draw() looks here.
        return 'Geometry Plot'

    @staticmethod
    def _position_label(geo_obj, label):
        """Return a (lon, lat) where the label of a polygon/line/point can be placed."""
        from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon, Polygon

        # A hash of the label is used in choosing a point along the polygon or line that
        # will be returned. This "psuedo-randomizes" the position of a label, in hopes of
        # spatially dispersing the labels and lessening the chance that labels overlap.
        label_hash = sum(map(ord, str(label)))

        # If object is a MultiPolygon or MultiLineString, associate the label with the single
        # largest Polygon or LineString from the collection. If MultiPoint, associate the label
        # with one of the Points in the MultiPoint, chosen based on the label hash.
        if isinstance(geo_obj, (MultiPolygon, MultiLineString)):
            geo_obj = max(geo_obj.geoms, key=lambda x: x.length)
        elif isinstance(geo_obj, MultiPoint):
            geo_obj = geo_obj.geoms[label_hash % len(geo_obj.geoms)]

        # Get the list of coordinates of the polygon/line/point
        coords = geo_obj.exterior.coords if isinstance(geo_obj, Polygon) else geo_obj.coords

        return coords[label_hash % len(coords)]

    def _draw_label(self, text, lon, lat, color='black', outline='white', offset=(0, 0)):
        """Draw a label to the plot.

        Parameters
        ----------
        text : str
            The label's text
        lon : float
            Longitude at which to position the label
        lat : float
            Latitude at which to position the label
        color : str (default: 'black')
            Name or hex code for the color of the text
        outline : str (default: 'white')
            Name or hex code of the color of the outline of the text
        offset : tuple (default: (0, 0))
            A tuple containing the x- and y-offset of the label, respectively
        """
        path_effects = [patheffects.withStroke(linewidth=4, foreground=outline)]
        self.parent.ax.add_artist(TextCollection([lon], [lat], [str(text)],
                                                 va='center',
                                                 ha='center',
                                                 offset=offset,
                                                 weight='demi',
                                                 size=self.label_fontsize,
                                                 color=color,
                                                 path_effects=path_effects,
                                                 transform=ccrs.PlateCarree()))

    def draw(self):
        """Draw the plot."""
        if self._need_redraw:
            if getattr(self, 'handles', None) is None:
                self._build()
            self._need_redraw = False

    def copy(self):
        """Return a copy of the plot."""
        return copy.copy(self)

    def _build(self):
        """Build the plot by calling needed plotting methods as necessary."""
        from shapely.geometry import (LineString, MultiLineString, MultiPoint, MultiPolygon,
                                      Point, Polygon)

        # Cast attributes to a list if None, since traitlets doesn't call validators (like
        # `_valid_color_list()` and `_valid_labels()`) when the proposed value is None.
        self.fill = ['none'] if self.fill is None else self.fill
        self.stroke = ['none'] if self.stroke is None else self.stroke
        self.labels = [''] if self.labels is None else self.labels
        self.label_edgecolor = (['none'] if self.label_edgecolor is None
                                else self.label_edgecolor)
        self.label_facecolor = (['none'] if self.label_facecolor is None
                                else self.label_facecolor)

        # Each Shapely object is plotted separately with its corresponding colors and label
        for geo_obj, stroke, strokewidth, fill, label, fontcolor, fontoutline in zip(
                self.geometry, cycle(self.stroke), cycle(self.stroke_width), cycle(self.fill),
                cycle(self.labels), cycle(self.label_facecolor), cycle(self.label_edgecolor)):
            kwargs = self.mpl_args.copy()
            # Plot the Shapely object with the appropriate method and colors
            if isinstance(geo_obj, (MultiPolygon, Polygon)):
                kwargs.setdefault('edgecolor', stroke)
                kwargs.setdefault('linewidths', strokewidth)
                kwargs.setdefault('facecolor', fill)
                kwargs.setdefault('crs', ccrs.PlateCarree())
                self.parent.ax.add_geometries([geo_obj], **kwargs)
            elif isinstance(geo_obj, (MultiLineString, LineString)):
                kwargs.setdefault('edgecolor', stroke)
                kwargs.setdefault('linewidths', strokewidth)
                kwargs.setdefault('facecolor', 'none')
                kwargs.setdefault('crs', ccrs.PlateCarree())
                self.parent.ax.add_geometries([geo_obj], **kwargs)
            elif isinstance(geo_obj, MultiPoint):
                kwargs.setdefault('color', fill)
                kwargs.setdefault('marker', self.marker)
                kwargs.setdefault('transform', ccrs.PlateCarree())
                for point in geo_obj.geoms:
                    lon, lat = point.coords[0]
                    self.parent.ax.plot(lon, lat, **kwargs)
            elif isinstance(geo_obj, Point):
                kwargs.setdefault('color', fill)
                kwargs.setdefault('marker', self.marker)
                kwargs.setdefault('transform', ccrs.PlateCarree())
                lon, lat = geo_obj.coords[0]
                self.parent.ax.plot(lon, lat, **kwargs)

            # Plot labels if provided
            if label:
                # If fontcolor is None/'none', choose a font color
                if fontcolor in [None, 'none'] and stroke not in [None, 'none']:
                    fontcolor = stroke
                elif fontcolor in [None, 'none']:
                    fontcolor = 'black'

                # If fontoutline is None/'none', choose a font outline
                if fontoutline in [None, 'none'] and fill not in [None, 'none']:
                    fontoutline = fill
                elif fontoutline in [None, 'none']:
                    fontoutline = 'white'

                # Choose a point along the polygon/line/point to place label
                lon, lat = self._position_label(geo_obj, label)

                # If polygon, put label directly on edge of polygon. If line or point, put
                # label slightly below line/point.
                offset = (0, 0) if isinstance(geo_obj, (MultiPolygon, Polygon)) else (0, -12)

                # Finally, draw the label
                self._draw_label(label, lon, lat, fontcolor, fontoutline, offset)
