# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Create Station-model plots."""

try:
    from enum import Enum
except ImportError:
    from enum34 import Enum

import matplotlib
import numpy as np

from .wx_symbols import (current_weather, high_clouds, low_clouds, mid_clouds,
                         pressure_tendency, sky_cover, wx_symbol_font)
from ..cbook import is_string_like
from ..package_tools import Exporter

exporter = Exporter(globals())


@exporter.export
class StationPlot(object):
    """Make a standard meteorological station plot.

    Plots values, symbols, or text spaced around a central location. Can also plot wind
    barbs as the center of the location.
    """

    location_names = {'C': (0, 0), 'N': (0, 1), 'NE': (1, 1), 'E': (1, 0), 'SE': (1, -1),
                      'S': (0, -1), 'SW': (-1, -1), 'W': (-1, 0), 'NW': (-1, 1)}

    def __init__(self, ax, x, y, fontsize=10, spacing=None, transform=None, **kwargs):
        """Initialize the StationPlot with items that do not change.

        This sets up the axes and station locations. The `fontsize` and `spacing`
        are also specified here to ensure that they are consistent between individual
        station elements.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The :class:`~matplotlib.axes.Axes` for plotting
        x : array_like
            The x location of the stations in the plot
        y : array_like
            The y location of the stations in the plot
        fontsize : int
            The fontsize to use for drawing text
        spacing : int
            The spacing, in points, that corresponds to a single increment between
            station plot elements.
        transform : matplotlib.transforms.Transform (or compatible)
            The default transform to apply to the x and y positions when plotting.
        kwargs
            Additional keyword arguments to use for matplotlib's plotting functions.
            These will be passed to all the plotting methods, and thus need to be valid
            for all plot types, such as `clip_on`.

        """
        self.ax = ax
        self.x = x
        self.y = y
        self.fontsize = fontsize
        self.spacing = fontsize if spacing is None else spacing
        self.transform = transform
        self.items = {}
        self.barbs = None
        self.default_kwargs = kwargs

    def plot_symbol(self, location, codes, symbol_mapper, **kwargs):
        """At the specified location in the station model plot a set of symbols.

        This specifies that at the offset `location`, the data in `codes` should be
        converted to unicode characters (for our :data:`wx_symbol_font`) using `symbol_mapper`,
        and plotted.

        Additional keyword arguments given will be passed onto the actual plotting
        code; this is useful for specifying things like color or font properties.

        If something has already been plotted at this location, it will be replaced.

        Parameters
        ----------
        location : str or tuple[float, float]
            The offset (relative to center) to plot this parameter. If str, should be one of
            'C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'. Otherwise, should be a tuple
            specifying the number of increments in the x and y directions; increments
            are multiplied by `spacing` to give offsets in x and y relative to the center.
        codes : array_like
            The numeric values that should be converted to unicode characters for plotting.
        symbol_mapper : callable
            Controls converting data values to unicode code points for the
            :data:`wx_symbol_font` font. This should take a value and return a single unicode
            character. See :mod:`metpy.plots.wx_symbols` for included mappers.
        kwargs
            Additional keyword arguments to use for matplotlib's plotting functions.


        .. plot::

           import matplotlib.pyplot as plt
           import numpy as np
           from math import ceil

           from metpy.plots import StationPlot
           from metpy.plots.wx_symbols import current_weather, current_weather_auto
           from metpy.plots.wx_symbols import low_clouds, mid_clouds, high_clouds
           from metpy.plots.wx_symbols import sky_cover, pressure_tendency


           def plot_symbols(mapper, name, nwrap=12, figsize=(10, 1.4)):

               # Determine how many symbols there are and layout in rows of nwrap
               # if there are more than nwrap symbols
               num_symbols = len(mapper)
               codes = np.arange(len(mapper))
               ncols = nwrap
               if num_symbols <= nwrap:
                   nrows = 1
                   x = np.linspace(0, 1, len(mapper))
                   y = np.ones_like(x)
                   ax_height = 0.8
               else:
                   nrows = int(ceil(num_symbols / ncols))
                   x = np.tile(np.linspace(0, 1, ncols), nrows)[:num_symbols]
                   y = np.repeat(np.arange(nrows, 0, -1), ncols)[:num_symbols]
                   figsize = (10, 1 * nrows + 0.4)
                   ax_height = 0.8 + 0.018 * nrows

               fig = plt.figure(figsize=figsize,  dpi=300)
               ax = fig.add_axes([0, 0, 1, ax_height])
               ax.set_title(name, size=20)
               ax.xaxis.set_ticks([])
               ax.yaxis.set_ticks([])
               ax.set_frame_on(False)

               # Plot
               sp = StationPlot(ax, x, y, fontsize=36)
               sp.plot_symbol('C', codes, mapper)
               sp.plot_parameter((0, -1), codes, fontsize=18)

               ax.set_ylim(-0.05, nrows + 0.5)

               plt.show()


           plot_symbols(current_weather, "Current Weather Symbols")
           plot_symbols(current_weather_auto, "Current Weather Auto Reported Symbols")
           plot_symbols(low_clouds, "Low Cloud Symbols")
           plot_symbols(mid_clouds, "Mid Cloud Symbols")
           plot_symbols(high_clouds, "High Cloud Symbols")
           plot_symbols(sky_cover, "Sky Cover Symbols")
           plot_symbols(pressure_tendency, "Pressure Tendency Symbols")

        See Also
        --------
        plot_barb, plot_parameter, plot_text

        """
        # Make sure we use our font for symbols
        kwargs['fontproperties'] = wx_symbol_font.copy()
        return self.plot_parameter(location, codes, symbol_mapper, **kwargs)

    def plot_parameter(self, location, parameter, formatter='.0f', **kwargs):
        """At the specified location in the station model plot a set of values.

        This specifies that at the offset `location`, the data in `parameter` should be
        plotted. The conversion of the data values to a string is controlled by `formatter`.

        Additional keyword arguments given will be passed onto the actual plotting
        code; this is useful for specifying things like color or font properties.

        If something has already been plotted at this location, it will be replaced.

        Parameters
        ----------
        location : str or tuple[float, float]
            The offset (relative to center) to plot this parameter. If str, should be one of
            'C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'. Otherwise, should be a tuple
            specifying the number of increments in the x and y directions; increments
            are multiplied by `spacing` to give offsets in x and y relative to the center.
        parameter : array_like
            The numeric values that should be plotted
        formatter : str or callable, optional
            How to format the data as a string for plotting. If a string, it should be
            compatible with the :func:`format` builtin. If a callable, this should take a
            value and return a string. Defaults to '0.f'.
        kwargs
            Additional keyword arguments to use for matplotlib's plotting functions.


        See Also
        --------
        plot_barb, plot_symbol, plot_text

        """
        text = self._to_string_list(parameter, formatter)
        return self.plot_text(location, text, **kwargs)

    def plot_text(self, location, text, **kwargs):
        """At the specified location in the station model plot a collection of text.

        This specifies that at the offset `location`, the strings in `text` should be
        plotted.

        Additional keyword arguments given will be passed onto the actual plotting
        code; this is useful for specifying things like color or font properties.

        If something has already been plotted at this location, it will be replaced.

        Parameters
        ----------
        location : str or tuple[float, float]
            The offset (relative to center) to plot this parameter. If str, should be one of
            'C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'. Otherwise, should be a tuple
            specifying the number of increments in the x and y directions; increments
            are multiplied by `spacing` to give offsets in x and y relative to the center.
        text : list (or array) of strings
            The strings that should be plotted
        kwargs
            Additional keyword arguments to use for matplotlib's plotting functions.

        See Also
        --------
        plot_barb, plot_parameter, plot_symbol

        """
        location = self._handle_location(location)

        kwargs = self._make_kwargs(kwargs)
        text_collection = self.ax.scattertext(self.x, self.y, text, loc=location,
                                              size=kwargs.pop('fontsize', self.fontsize),
                                              **kwargs)
        if location in self.items:
            self.items[location].remove()
        self.items[location] = text_collection
        return text_collection

    def plot_barb(self, u, v, **kwargs):
        r"""At the center of the station model plot wind barbs.

        Additional keyword arguments given will be passed onto matplotlib's
        :meth:`~matplotlib.axes.Axes.barbs` function; this is useful for specifying things
        like color or line width.

        Parameters
        ----------
        u : array-like
            The data to use for the u-component of the barbs.
        v : array-like
            The data to use for the v-component of the barbs.
        plot_units: `pint.unit`
            Units to plot in (performing conversion if necessary). Defaults to given units.
        kwargs
            Additional keyword arguments to pass to matplotlib's
            :meth:`~matplotlib.axes.Axes.barbs` function.

        See Also
        --------
        plot_parameter, plot_symbol, plot_text

        """
        kwargs = self._make_kwargs(kwargs)

        # If plot_units specified, convert the data to those units
        plotting_units = kwargs.pop('plot_units', None)
        if plotting_units:
            if hasattr(u, 'units') and hasattr(v, 'units'):
                u = u.to(plotting_units)
                v = v.to(plotting_units)
            else:
                raise ValueError('To convert to plotting units, units must be attached to '
                                 'u and v wind components.')

        # Strip units, CartoPy transform doesn't like
        u = np.array(u)
        v = np.array(v)

        # Empirically determined
        pivot = 0.51 * np.sqrt(self.fontsize)
        length = 1.95 * np.sqrt(self.fontsize)
        defaults = {'sizes': {'spacing': .15, 'height': 0.5, 'emptybarb': 0.35},
                    'length': length, 'pivot': pivot}
        defaults.update(kwargs)

        # Remove old barbs
        if self.barbs:
            self.barbs.remove()

        # Handle transforming our center points. CartoPy doesn't like 1D barbs
        # TODO: This can be removed for cartopy > 0.14.3
        if hasattr(self.ax, 'projection') and 'transform' in kwargs:
            trans = kwargs['transform']
            try:
                kwargs['transform'] = trans._as_mpl_transform(self.ax)
            except AttributeError:
                pass
            u, v = self.ax.projection.transform_vectors(trans, self.x, self.y, u, v)

            # Since we've re-implemented CartoPy's barbs, we need to skip calling it here
            self.barbs = matplotlib.axes.Axes.barbs(self.ax, self.x, self.y, u, v, **defaults)
        else:
            self.barbs = self.ax.barbs(self.x, self.y, u, v, **defaults)

    def _make_kwargs(self, kwargs):
        """Assemble kwargs as necessary.

        Inserts our defaults as well as ensures transform is present when appropriate.
        """
        # Use default kwargs and update with additional ones
        all_kw = self.default_kwargs.copy()
        all_kw.update(kwargs)

        # Pass transform if necessary
        if 'transform' not in all_kw and self.transform:
            all_kw['transform'] = self.transform

        return all_kw

    @staticmethod
    def _to_string_list(vals, fmt):
        """Convert a sequence of values to a list of strings."""
        if not callable(fmt):
            def formatter(s):
                """Turn a format string into a callable."""
                if hasattr(s, 'units'):
                    s = s.item()
                return format(s, fmt)
        else:
            formatter = fmt

        return [formatter(v) if np.isfinite(v) else '' for v in vals]

    def _handle_location(self, location):
        """Process locations to get a consistent set of tuples for location."""
        if is_string_like(location):
            location = self.location_names[location]
        xoff, yoff = location
        return xoff * self.spacing, yoff * self.spacing


@exporter.export
class StationPlotLayout(dict):
    r"""make a layout to encapsulate plotting using :class:`StationPlot`.

    This class keeps a collection of offsets, plot formats, etc. for a parameter based
    on its name. This then allows a dictionary of data (or any object that allows looking
    up of arrays based on a name) to be passed to :meth:`plot()` to plot the data all at once.

    See Also
    --------
    StationPlot

    """

    class PlotTypes(Enum):
        r"""Different plotting types for the layout.

        Controls how items are displayed (e.g. converting values to symbols).
        """

        value = 1
        symbol = 2
        text = 3
        barb = 4

    def add_value(self, location, name, fmt='.0f', units=None, **kwargs):
        r"""Add a numeric value to the station layout.

        This specifies that at the offset `location`, data should be pulled from the data
        container using the key `name` and plotted. The conversion of the data values to
        a string is controlled by `fmt`. The units required for plotting can also
        be passed in using `units`, which will cause the data to be converted before
        plotting.

        Additional keyword arguments given will be passed onto the actual plotting
        code; this is useful for specifying things like color or font properties.

        Parameters
        ----------
        location : str or tuple[float, float]
            The offset (relative to center) to plot this value. If str, should be one of
            'C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'. Otherwise, should be a tuple
            specifying the number of increments in the x and y directions.
        name : str
            The name of the parameter, which is used as a key to pull data out of the
            data container passed to :meth:`plot`.
        fmt : str or callable, optional
            How to format the data as a string for plotting. If a string, it should be
            compatible with the :func:`format` builtin. If a callable, this should take a
            value and return a string. Defaults to '0.f'.
        units : pint-compatible unit, optional
            The units to use for plotting. Data will be converted to this unit before
            conversion to a string. If not specified, no conversion is done.
        kwargs
            Additional keyword arguments to use for matplotlib's plotting functions.

        See Also
        --------
        add_barb, add_symbol, add_text

        """
        self[location] = (self.PlotTypes.value, name, (fmt, units, kwargs))

    def add_symbol(self, location, name, symbol_mapper, **kwargs):
        r"""Add a symbol to the station layout.

        This specifies that at the offset `location`, data should be pulled from the data
        container using the key `name` and plotted. Data values will converted to glyphs
        appropriate for MetPy's symbol font using the callable `symbol_mapper`.

        Additional keyword arguments given will be passed onto the actual plotting
        code; this is useful for specifying things like color or font properties.

        Parameters
        ----------
        location : str or tuple[float, float]
            The offset (relative to center) to plot this value. If str, should be one of
            'C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'. Otherwise, should be a tuple
            specifying the number of increments in the x and y directions.
        name : str
            The name of the parameter, which is used as a key to pull data out of the
            data container passed to :meth:`plot`.
        symbol_mapper : callable
            Controls converting data values to unicode code points for the
            :data:`wx_symbol_font` font. This should take a value and return a single unicode
            character. See :mod:`metpy.plots.wx_symbols` for included mappers.
        kwargs
            Additional keyword arguments to use for matplotlib's plotting functions.

        See Also
        --------
        add_barb, add_text, add_value

        """
        self[location] = (self.PlotTypes.symbol, name, (symbol_mapper, kwargs))

    def add_text(self, location, name, **kwargs):
        r"""Add a text field to the  station layout.

        This specifies that at the offset `location`, data should be pulled from the data
        container using the key `name` and plotted directly as text with no conversion
        applied.

        Additional keyword arguments given will be passed onto the actual plotting
        code; this is useful for specifying things like color or font properties.

        Parameters
        ----------
        location : str or tuple(float, float)
            The offset (relative to center) to plot this value. If str, should be one of
            'C', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'. Otherwise, should be a tuple
            specifying the number of increments in the x and y directions.
        name : str
            The name of the parameter, which is used as a key to pull data out of the
            data container passed to :meth:`plot`.
        kwargs
            Additional keyword arguments to use for matplotlib's plotting functions.

        See Also
        --------
        add_barb, add_symbol, add_value

        """
        self[location] = (self.PlotTypes.text, name, kwargs)

    def add_barb(self, u_name, v_name, units=None, **kwargs):
        r"""Add a wind barb to the center of the station layout.

        This specifies that u- and v-component data should be pulled from the data
        container using the keys `u_name` and `v_name`, respectively, and plotted as
        a wind barb at the center of the station plot. If `units` are given, both
        components will be converted to these units.

        Additional keyword arguments given will be passed onto the actual plotting
        code; this is useful for specifying things like color or line width.

        Parameters
        ----------
        u_name : str
            The name of the parameter for the u-component for `barbs`, which is used as
            a key to pull data out of the data container passed to :meth:`plot`.
        v_name : str
            The name of the parameter for the v-component for `barbs`, which is used as
            a key to pull data out of the data container passed to :meth:`plot`.
        units : pint-compatible unit, optional
            The units to use for plotting. Data will be converted to this unit before
            conversion to a string. If not specified, no conversion is done.
        kwargs
            Additional keyword arguments to use for matplotlib's
            :meth:`~matplotlib.axes.Axes.barbs` function.

        See Also
        --------
        add_symbol, add_text, add_value

        """
        # Not sure if putting the v_name as a plot-specific option is appropriate,
        # but it seems simpler than making name code in plot handle tuples
        self['barb'] = (self.PlotTypes.barb, (u_name, v_name), (units, kwargs))

    def names(self):
        """Get the list of names used by the layout.

        Returns
        -------
        list[str]
            the list of names of variables used by the layout

        """
        ret = []
        for item in self.values():
            if item[0] == self.PlotTypes.barb:
                ret.extend(item[1])
            else:
                ret.append(item[1])
        return ret

    def plot(self, plotter, data_dict):
        """Plot a collection of data using this layout for a station plot.

        This function iterates through the entire specified layout, pulling the fields named
        in the layout from `data_dict` and plotting them using `plotter` as specified
        in the layout. Fields present in the layout, but not in `data_dict`, are ignored.

        Parameters
        ----------
        plotter : StationPlot
            :class:`StationPlot` to use to plot the data. This controls the axes,
            spacing, station locations, etc.
        data_dict : dict[str, array-like]
            Data container that maps a name to an array of data. Data from this object
            will be used to fill out the station plot.

        """
        def coerce_data(dat, u):
            try:
                return dat.to(u).magnitude
            except AttributeError:
                return dat

        for loc, info in self.items():
            typ, name, args = info
            if typ == self.PlotTypes.barb:
                # Try getting the data
                u_name, v_name = name
                u_data = data_dict.get(u_name)
                v_data = data_dict.get(v_name)

                # Plot if we have the data
                if not (v_data is None or u_data is None):
                    units, kwargs = args
                    plotter.plot_barb(coerce_data(u_data, units), coerce_data(v_data, units),
                                      **kwargs)
            else:
                # Check that we have the data for this location
                data = data_dict.get(name)
                if data is not None:
                    # If we have it, hand it to the appropriate method
                    if typ == self.PlotTypes.value:
                        fmt, units, kwargs = args
                        plotter.plot_parameter(loc, coerce_data(data, units), fmt, **kwargs)
                    elif typ == self.PlotTypes.symbol:
                        mapper, kwargs = args
                        plotter.plot_symbol(loc, data, mapper, **kwargs)
                    elif typ == self.PlotTypes.text:
                        plotter.plot_text(loc, data, **args)

    def __repr__(self):
        """Return string representation of layout."""
        return ('{'
                + ', '.join('{0}: ({1[0].name}, {1[1]}, ...)'.format(loc, info)
                            for loc, info in sorted(self.items()))
                + '}')


with exporter:
    #: :desc: Simple station plot layout
    simple_layout = StationPlotLayout()
    simple_layout.add_barb('eastward_wind', 'northward_wind', 'knots')
    simple_layout.add_value('NW', 'air_temperature', units='degC')
    simple_layout.add_value('SW', 'dew_point_temperature', units='degC')
    simple_layout.add_value('NE', 'air_pressure_at_sea_level', units='mbar',
                            fmt=lambda v: format(10 * v, '03.0f')[-3:])
    simple_layout.add_symbol('C', 'cloud_coverage', sky_cover)
    simple_layout.add_symbol('W', 'present_weather', current_weather)

    #: Full NWS station plot `layout`__
    #:
    #: __ http://oceanservice.noaa.gov/education/yos/resource/JetStream/synoptic/wxmaps.htm
    nws_layout = StationPlotLayout()
    nws_layout.add_value((-1, 1), 'air_temperature', units='degF')
    nws_layout.add_symbol((0, 2), 'high_cloud_type', high_clouds)
    nws_layout.add_symbol((0, 1), 'medium_cloud_type', mid_clouds)
    nws_layout.add_symbol((0, -1), 'low_cloud_type', low_clouds)
    nws_layout.add_value((1, 1), 'air_pressure_at_sea_level', units='mbar',
                         fmt=lambda v: format(10 * v, '03.0f')[-3:])
    nws_layout.add_value((-2, 0), 'visibility_in_air', fmt='.0f', units='miles')
    nws_layout.add_symbol((-1, 0), 'present_weather', current_weather)
    nws_layout.add_symbol((0, 0), 'cloud_coverage', sky_cover)
    nws_layout.add_value((1, 0), 'tendency_of_air_pressure', units='mbar',
                         fmt=lambda v: ('-' if v < 0 else '') + format(10 * abs(v), '02.0f'))
    nws_layout.add_symbol((2, 0), 'tendency_of_air_pressure_symbol', pressure_tendency)
    nws_layout.add_barb('eastward_wind', 'northward_wind', units='knots')
    nws_layout.add_value((-1, -1), 'dew_point_temperature', units='degF')

    # TODO: Fix once we have the past weather symbols converted
    nws_layout.add_symbol((1, -1), 'past_weather', current_weather)
