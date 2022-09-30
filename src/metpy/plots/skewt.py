# Copyright (c) 2014,2015,2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Make Skew-T Log-P based plots.

Contain tools for making Skew-T Log-P plots, including the base plotting class,
`SkewT`, as well as a class for making a `Hodograph`.
"""

from contextlib import ExitStack

from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
from matplotlib.ticker import MultipleLocator, NullFormatter, ScalarFormatter
import matplotlib.transforms as transforms
import numpy as np

from ._util import colored_line
from ..calc import dewpoint, dry_lapse, el, lcl, moist_lapse, vapor_pressure
from ..calc.tools import _delete_masked_points
from ..interpolate import interpolate_1d
from ..package_tools import Exporter
from ..units import concatenate, units

exporter = Exporter(globals())


class SkewTTransform(transforms.Affine2D):
    """Perform Skew transform for Skew-T plotting.

    This works in pixel space, so is designed to be applied after the normal plotting
    transformations.
    """

    def __init__(self, bbox, rot):
        """Initialize skew transform.

        This needs a reference to the parent bounding box to do the appropriate math and
        to register it as a child so that the transform is invalidated and regenerated if
        the bounding box changes.
        """
        super().__init__()
        self._bbox = bbox
        self.set_children(bbox)
        self.invalidate()

        # We're not trying to support changing the rotation, so go ahead and convert to
        # the right factor for skewing here and just save that.
        self._rot_factor = np.tan(np.deg2rad(rot))

    def get_matrix(self):
        """Return transformation matrix."""
        if self._invalid:
            # The following matrix is equivalent to the following:
            # x0, y0 = self._bbox.xmin, self._bbox.ymin
            # self.translate(-x0, -y0).skew_deg(self._rot, 0).translate(x0, y0)
            # Setting it this way is just more efficient.
            self._mtx = np.array([[1.0, self._rot_factor, -self._rot_factor * self._bbox.ymin],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])

            # Need to clear both the invalid flag *and* reset the inverse, which is cached
            # by the parent class.
            self._invalid = 0
            self._inverted = None
        return self._mtx


class SkewXTick(maxis.XTick):
    r"""Make x-axis ticks for Skew-T plots.

    This class adds to the standard :class:`matplotlib.axis.XTick` dynamic checking
    for whether a top or bottom tick is actually within the data limits at that part
    and draw as appropriate. It also performs similar checking for gridlines.
    """

    # Taken from matplotlib's SkewT example to update for matplotlib 3.1's changes to
    # state management for ticks. See matplotlib/matplotlib#10088
    def draw(self, renderer):
        """Draw the tick."""
        # When adding the callbacks with `stack.callback`, we fetch the current
        # visibility state of the artist with `get_visible`; the ExitStack will
        # restore these states (`set_visible`) at the end of the block (after
        # the draw).
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line,
                           self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())

            self.tick1line.set_visible(self.tick1line.get_visible() and self.lower_in_bounds)
            self.label1.set_visible(self.label1.get_visible() and self.lower_in_bounds)
            self.tick2line.set_visible(self.tick2line.get_visible() and self.upper_in_bounds)
            self.label2.set_visible(self.label2.get_visible() and self.upper_in_bounds)
            self.gridline.set_visible(self.gridline.get_visible() and self.grid_in_bounds)
            super().draw(renderer)

    @property
    def lower_in_bounds(self):
        """Whether the lower part of the tick is in bounds."""
        return transforms.interval_contains(self.axes.lower_xlim, self.get_loc())

    @property
    def upper_in_bounds(self):
        """Whether the upper part of the tick is in bounds."""
        return transforms.interval_contains(self.axes.upper_xlim, self.get_loc())

    @property
    def grid_in_bounds(self):
        """Whether any of the tick grid line is in bounds."""
        return transforms.interval_contains(self.axes.xaxis.get_view_interval(),
                                            self.get_loc())


class SkewXAxis(maxis.XAxis):
    r"""Make an x-axis that works properly for Skew-T plots.

    This class exists to force the use of our custom :class:`SkewXTick` as well
    as provide a custom value for interval that combines the extents of the
    upper and lower x-limits from the axes.
    """

    def _get_tick(self, major):
        return SkewXTick(self.axes, None, major=major)

    # Needed to properly handle tight bbox
    def _get_ticklabel_bboxes(self, ticks, renderer):
        """Return lists of bboxes for ticks' label1's and label2's."""
        return ([tick.label1.get_window_extent(renderer)
                 for tick in ticks if tick.label1.get_visible() and tick.lower_in_bounds],
                [tick.label2.get_window_extent(renderer)
                 for tick in ticks if tick.label2.get_visible() and tick.upper_in_bounds])

    # Older name used on Matplotlib < 3.6
    _get_tick_bboxes = _get_ticklabel_bboxes

    def get_view_interval(self):
        """Get the view interval."""
        return self.axes.upper_xlim[0], self.axes.lower_xlim[1]


class SkewSpine(mspines.Spine):
    r"""Make an x-axis spine that works properly for Skew-T plots.

    This class exists to use the separate x-limits from the axes to properly
    locate the spine.
    """

    def _adjust_location(self):
        pts = self._path.vertices
        if self.spine_type == 'top':
            pts[:, 0] = self.axes.upper_xlim
        else:
            pts[:, 0] = self.axes.lower_xlim


class SkewXAxes(Axes):
    r"""Make a set of axes for Skew-T plots.

    This class handles registration of the skew-xaxes as a projection as well as setting up
    the appropriate transformations. It also makes sure we use our instances for spines
    and x-axis: :class:`SkewSpine` and :class:`SkewXAxis`. It provides properties to
    facilitate finding the x-limits for the bottom and top of the plot as well.
    """

    # The projection must specify a name.  This will be used be the
    # user to select the projection, i.e. ``subplot(111,
    # projection='skewx')``.
    name = 'skewx'

    def __init__(self, *args, **kwargs):
        r"""Initialize `SkewXAxes`.

        Parameters
        ----------
        args : Arbitrary positional arguments
            Passed to :class:`matplotlib.axes.Axes`

        position: int, optional
            The rotation of the x-axis against the y-axis, in degrees.

        kwargs : Arbitrary keyword arguments
            Passed to :class:`matplotlib.axes.Axes`

        """
        # This needs to be popped and set before moving on
        self.rot = kwargs.pop('rotation', 30)
        super().__init__(*args, **kwargs)

    def _init_axis(self):
        # Taken from Axes and modified to use our modified X-axis
        self.xaxis = SkewXAxis(self)
        self.spines['top'].register_axis(self.xaxis)
        self.spines['bottom'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.spines['left'].register_axis(self.yaxis)
        self.spines['right'].register_axis(self.yaxis)

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        # pylint: disable=unused-argument
        return {'top': SkewSpine.linear_spine(self, 'top'),
                'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                'left': mspines.Spine.linear_spine(self, 'left'),
                'right': mspines.Spine.linear_spine(self, 'right')}

    def _set_lim_and_transforms(self):
        """Set limits and transforms.

        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.

        """
        # Get the standard transform setup from the Axes base class
        super()._set_lim_and_transforms()

        # This transformation handles the skewing
        skew_trans = SkewTTransform(self.bbox, self.rot)

        # Create the full transform from Data to Pixels
        self.transData += skew_trans

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.
        self._xaxis_transform += skew_trans

    @property
    def lower_xlim(self):
        """Get the data limits for the x-axis along the bottom of the axes."""
        return self.axes.viewLim.intervalx

    @property
    def upper_xlim(self):
        """Get the data limits for the x-axis along the top of the axes."""
        return self.transData.inverted().transform([[self.bbox.xmin, self.bbox.ymax],
                                                    self.bbox.max])[:, 0]


# Now register the projection with matplotlib so the user can select it.
register_projection(SkewXAxes)


@exporter.export
class SkewT:
    r"""Make Skew-T log-P plots of data.

    This class simplifies the process of creating Skew-T log-P plots in
    using matplotlib. It handles requesting the appropriate skewed projection,
    and provides simplified wrappers to make it easy to plot data, add wind
    barbs, and add other lines to the plots (e.g. dry adiabats)

    Attributes
    ----------
    ax : `matplotlib.axes.Axes`
        The underlying Axes instance, which can be used for calling additional
        plot functions (e.g. `axvline`)

    """

    def __init__(self, fig=None, rotation=30, subplot=None, rect=None, aspect=80.5):
        r"""Create SkewT - logP plots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Source figure to use for plotting. If none is given, a new
            :class:`matplotlib.figure.Figure` instance will be created.
        rotation : float or int, optional
            Controls the rotation of temperature relative to horizontal. Given
            in degrees counterclockwise from x-axis. Defaults to 30 degrees.
        subplot : tuple[int, int, int] or `matplotlib.gridspec.SubplotSpec` instance, optional
            Controls the size/position of the created subplot. This allows creating
            the skewT as part of a collection of subplots. If subplot is a tuple, it
            should conform to the specification used for
            :meth:`matplotlib.figure.Figure.add_subplot`. The
            :class:`matplotlib.gridspec.SubplotSpec`
            can be created by using :class:`matplotlib.gridspec.GridSpec`.
        rect : tuple[float, float, float, float], optional
            Rectangle (left, bottom, width, height) in which to place the axes. This
            allows the user to place the axes at an arbitrary point on the figure.
        aspect : float, int, or Literal['auto'], optional
            Aspect ratio (i.e. ratio of y-scale to x-scale) to maintain in the plot.
            Defaults to 80.5. Passing the string ``'auto'`` tells matplotlib to handle
            the aspect ratio automatically (this is not recommended for SkewT).

        """
        if fig is None:
            import matplotlib.pyplot as plt
            figsize = plt.rcParams.get('figure.figsize', (7, 7))
            fig = plt.figure(figsize=figsize)
        self._fig = fig

        if rect and subplot:
            raise ValueError("Specify only one of `rect' and `subplot', but not both")

        elif rect:
            self.ax = fig.add_axes(rect, projection='skewx', rotation=rotation)

        else:
            if subplot is not None:
                # Handle being passed a tuple for the subplot, or a GridSpec instance
                try:
                    len(subplot)
                except TypeError:
                    subplot = (subplot,)
            else:
                subplot = (1, 1, 1)

            self.ax = fig.add_subplot(*subplot, projection='skewx', rotation=rotation)

        # Set the yaxis as inverted with log scaling
        self.ax.set_yscale('log')

        # Override default ticking for log scaling
        self.ax.yaxis.set_major_formatter(ScalarFormatter())
        self.ax.yaxis.set_major_locator(MultipleLocator(100))
        self.ax.yaxis.set_minor_formatter(NullFormatter())

        # Needed to make sure matplotlib doesn't freak out and create a bunch of ticks
        # Also takes care of inverting the y-axis
        self.ax.set_ylim(1050, 100)
        self.ax.yaxis.set_units(units.hPa)

        # Try to make sane default temperature plotting ticks
        self.ax.xaxis.set_major_locator(MultipleLocator(10))
        self.ax.xaxis.set_units(units.degC)
        self.ax.set_xlim(-40, 50)
        self.ax.grid(True)

        self.mixing_lines = None
        self.dry_adiabats = None
        self.moist_adiabats = None

        # Maintain a reasonable ratio of data limits.
        self.ax.set_aspect(aspect, adjustable='box')

    def plot(self, pressure, t, *args, **kwargs):
        r"""Plot data.

        Simple wrapper around plot so that pressure is the first (independent)
        input. This is essentially a wrapper around `plot`.

        Parameters
        ----------
        pressure : array-like
            pressure values
        t : array-like
            temperature values, can also be used for things like dew point
        args
            Other positional arguments to pass to :func:`~matplotlib.pyplot.plot`
        kwargs
            Other keyword arguments to pass to :func:`~matplotlib.pyplot.plot`

        Returns
        -------
        list[matplotlib.lines.Line2D]
            lines plotted

        See Also
        --------
        :func:`matplotlib.pyplot.plot`

        """
        # Skew-T logP plotting
        t, pressure = _delete_masked_points(t, pressure)
        return self.ax.plot(t, pressure, *args, **kwargs)

    def plot_barbs(self, pressure, u, v, c=None, xloc=1.0, x_clip_radius=0.1,
                   y_clip_radius=0.08, **kwargs):
        r"""Plot wind barbs.

        Adds wind barbs to the skew-T plot. This is a wrapper around the
        `barbs` command that adds to appropriate transform to place the
        barbs in a vertical line, located as a function of pressure.

        Parameters
        ----------
        pressure : array-like
            pressure values
        u : array-like
            U (East-West) component of wind
        v : array-like
            V (North-South) component of wind
        c : array-like, optional
            An optional array used to map colors to the barbs
        xloc : float, optional
            Position for the barbs, in normalized axes coordinates, where 0.0
            denotes far left and 1.0 denotes far right. Defaults to far right.
        x_clip_radius : float, optional
            Space, in normalized axes coordinates, to leave before clipping
            wind barbs in the x-direction. Defaults to 0.1.
        y_clip_radius : float, optional
            Space, in normalized axes coordinates, to leave above/below plot
            before clipping wind barbs in the y-direction. Defaults to 0.08.
        plot_units: `pint.Unit`
            Units to plot in (performing conversion if necessary). Defaults to given units.
        kwargs
            Other keyword arguments to pass to :func:`~matplotlib.pyplot.barbs`

        Returns
        -------
        matplotlib.quiver.Barbs
            instance created

        See Also
        --------
        :func:`matplotlib.pyplot.barbs`

        """
        # If plot_units specified, convert the data to those units
        plotting_units = kwargs.pop('plot_units', None)
        if plotting_units:
            if hasattr(u, 'units') and hasattr(v, 'units'):
                u = u.to(plotting_units)
                v = v.to(plotting_units)
            else:
                raise ValueError('To convert to plotting units, units must be attached to '
                                 'u and v wind components.')

        # Assemble array of x-locations in axes space
        x = np.empty_like(pressure)
        x.fill(xloc)

        # Do barbs plot at this location
        if c is not None:
            b = self.ax.barbs(x, pressure, u, v, c,
                              transform=self.ax.get_yaxis_transform(which='tick2'),
                              clip_on=True, zorder=2, **kwargs)
        else:
            b = self.ax.barbs(x, pressure, u, v,
                              transform=self.ax.get_yaxis_transform(which='tick2'),
                              clip_on=True, zorder=2, **kwargs)

        # Override the default clip box, which is the axes rectangle, so we can have
        # barbs that extend outside.
        ax_bbox = transforms.Bbox([[xloc - x_clip_radius, -y_clip_radius],
                                   [xloc + x_clip_radius, 1.0 + y_clip_radius]])
        b.set_clip_box(transforms.TransformedBbox(ax_bbox, self.ax.transAxes))
        return b

    def plot_dry_adiabats(self, t0=None, pressure=None, **kwargs):
        r"""Plot dry adiabats.

        Adds dry adiabats (lines of constant potential temperature) to the
        plot. The default style of these lines is dashed red lines with an alpha
        value of 0.5. These can be overridden using keyword arguments.

        Parameters
        ----------
        t0 : array-like, optional
            Starting temperature values in Kelvin. If none are given, they will be
            generated using the current temperature range at the bottom of
            the plot.
        pressure : array-like, optional
            Pressure values to be included in the dry adiabats. If not
            specified, they will be linearly distributed across the current
            plotted pressure range.
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.LineCollection`

        Returns
        -------
        matplotlib.collections.LineCollection
            instance created

        See Also
        --------
        :func:`~metpy.calc.dry_lapse`
        :meth:`plot_moist_adiabats`
        :class:`matplotlib.collections.LineCollection`

        """
        # Remove old lines
        if self.dry_adiabats:
            self.dry_adiabats.remove()

        # Determine set of starting temps if necessary
        if t0 is None:
            xmin, xmax = self.ax.get_xlim()
            t0 = units.Quantity(np.arange(xmin, xmax + 1, 10), 'degC')

        # Get pressure levels based on ylims if necessary
        if pressure is None:
            pressure = units.Quantity(np.linspace(*self.ax.get_ylim()), 'mbar')

        # Assemble into data for plotting
        t = dry_lapse(pressure, t0[:, np.newaxis],
                      units.Quantity(1000., 'mbar')).to(units.degC)
        linedata = [np.vstack((ti.m, pressure.m)).T for ti in t]

        # Add to plot
        kwargs.setdefault('colors', 'r')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('zorder', Line2D.zorder - 0.001)
        self.dry_adiabats = self.ax.add_collection(LineCollection(linedata, **kwargs))
        return self.dry_adiabats

    def plot_moist_adiabats(self, t0=None, pressure=None, **kwargs):
        r"""Plot moist adiabats.

        Adds saturated pseudo-adiabats (lines of constant equivalent potential
        temperature) to the plot. The default style of these lines is dashed
        blue lines with an alpha value of 0.5. These can be overridden using
        keyword arguments.

        Parameters
        ----------
        t0 : array-like, optional
            Starting temperature values in Kelvin. If none are given, they will be
            generated using the current temperature range at the bottom of
            the plot.
        pressure : array-like, optional
            Pressure values to be included in the moist adiabats. If not
            specified, they will be linearly distributed across the current
            plotted pressure range.
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.LineCollection`

        Returns
        -------
        matplotlib.collections.LineCollection
            instance created

        See Also
        --------
        :func:`~metpy.calc.moist_lapse`
        :meth:`plot_dry_adiabats`
        :class:`matplotlib.collections.LineCollection`

        """
        # Remove old lines
        if self.moist_adiabats:
            self.moist_adiabats.remove()

        # Determine set of starting temps if necessary
        if t0 is None:
            xmin, xmax = self.ax.get_xlim()
            t0 = units.Quantity(np.concatenate((np.arange(xmin, 0, 10),
                                                np.arange(0, xmax + 1, 5))), 'degC')

        # Get pressure levels based on ylims if necessary
        if pressure is None:
            pressure = units.Quantity(np.linspace(*self.ax.get_ylim()), 'mbar')

        # Assemble into data for plotting
        t = moist_lapse(pressure, t0, units.Quantity(1000., 'mbar')).to(units.degC)
        linedata = [np.vstack((ti.m, pressure.m)).T for ti in t]

        # Add to plot
        kwargs.setdefault('colors', 'b')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.5)
        kwargs.setdefault('zorder', Line2D.zorder - 0.001)
        self.moist_adiabats = self.ax.add_collection(LineCollection(linedata, **kwargs))
        return self.moist_adiabats

    def plot_mixing_lines(self, mixing_ratio=None, pressure=None, **kwargs):
        r"""Plot lines of constant mixing ratio.

        Adds lines of constant mixing ratio (isohumes) to the
        plot. The default style of these lines is dashed green lines with an
        alpha value of 0.8. These can be overridden using keyword arguments.

        Parameters
        ----------
        mixing_ratio : array-like, optional
            Unitless mixing ratio values to plot. If none are given, default
            values are used.
        pressure : array-like, optional
            Pressure values to be included in the isohumes. If not
            specified, they will be linearly distributed across the current
            plotted pressure range up to 600 mb.
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.LineCollection`

        Returns
        -------
        matplotlib.collections.LineCollection
            instance created

        See Also
        --------
        :class:`matplotlib.collections.LineCollection`

        """
        # Remove old lines
        if self.mixing_lines:
            self.mixing_lines.remove()

        # Default mixing level values if necessary
        if mixing_ratio is None:
            mixing_ratio = np.array([0.0004, 0.001, 0.002, 0.004, 0.007, 0.01,
                                     0.016, 0.024, 0.032])
        mixing_ratio = mixing_ratio.reshape(-1, 1)

        # Set pressure range if necessary
        if pressure is None:
            pressure = units.Quantity(np.linspace(600, max(self.ax.get_ylim())), 'mbar')

        # Assemble data for plotting
        td = dewpoint(vapor_pressure(pressure, mixing_ratio))
        linedata = [np.vstack((t.m, pressure.m)).T for t in td]

        # Add to plot
        kwargs.setdefault('colors', 'g')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.8)
        kwargs.setdefault('zorder', Line2D.zorder - 0.001)
        self.mixing_lines = self.ax.add_collection(LineCollection(linedata, **kwargs))
        return self.mixing_lines

    def shade_area(self, y, x1, x2=0, which='both', **kwargs):
        r"""Shade area between two curves.

        Shades areas between curves. Area can be where one is greater or less than the other
        or all areas shaded.

        Parameters
        ----------
        y : array-like
            1-dimensional array of numeric y-values
        x1 : array-like
            1-dimensional array of numeric x-values
        x2 : array-like
            1-dimensional array of numeric x-values
        which : str
            Specifies if `positive`, `negative`, or `both` areas are being shaded.
            Will be overridden by where.
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.PolyCollection`

        Returns
        -------
        :class:`matplotlib.collections.PolyCollection`

        See Also
        --------
        :class:`matplotlib.collections.PolyCollection`
        :meth:`matplotlib.axes.Axes.fill_betweenx`

        """
        fill_properties = {'positive':
                           {'facecolor': 'tab:red', 'alpha': 0.4, 'where': x1 > x2},
                           'negative':
                           {'facecolor': 'tab:blue', 'alpha': 0.4, 'where': x1 < x2},
                           'both':
                           {'facecolor': 'tab:green', 'alpha': 0.4, 'where': None}}

        try:
            fill_args = fill_properties[which]
            fill_args.update(kwargs)
        except KeyError:
            raise ValueError(f'Unknown option for which: {which}') from None

        arrs = y, x1, x2

        if fill_args['where'] is not None:
            arrs = arrs + (fill_args['where'],)
            fill_args.pop('where', None)

        fill_args['interpolate'] = True

        arrs = _delete_masked_points(*arrs)

        return self.ax.fill_betweenx(*arrs, **fill_args)

    def shade_cape(self, pressure, t, t_parcel, **kwargs):
        r"""Shade areas of Convective Available Potential Energy (CAPE).

        Shades areas where the parcel is warmer than the environment (areas of positive
        buoyancy.

        Parameters
        ----------
        pressure : array-like
            Pressure values
        t : array-like
            Temperature values
        t_parcel : array-like
            Parcel path temperature values
        limit_shading : bool
            Eliminate shading below the LCL or above the EL, default is True
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.PolyCollection`

        Returns
        -------
        :class:`matplotlib.collections.PolyCollection`

        See Also
        --------
        :class:`matplotlib.collections.PolyCollection`
        :meth:`matplotlib.axes.Axes.fill_betweenx`

        """
        return self.shade_area(pressure, t_parcel, t, which='positive', **kwargs)

    def shade_cin(self, pressure, t, t_parcel, dewpoint=None, **kwargs):
        r"""Shade areas of Convective INhibition (CIN).

        Shades areas where the parcel is cooler than the environment (areas of negative
        buoyancy). If `dewpoint` is passed in, negative area below the lifting condensation
        level or above the equilibrium level is not shaded.

        Parameters
        ----------
        pressure : array-like
            Pressure values
        t : array-like
            Temperature values
        t_parcel : array-like
            Parcel path temperature values
        dewpoint : array-like
            Dew point values, optional
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.PolyCollection`

        Returns
        -------
        :class:`matplotlib.collections.PolyCollection`

        See Also
        --------
        :class:`matplotlib.collections.PolyCollection`
        :meth:`matplotlib.axes.Axes.fill_betweenx`

        """
        if dewpoint is not None:
            lcl_p, _ = lcl(pressure[0], t[0], dewpoint[0])
            el_p, _ = el(pressure, t, dewpoint, t_parcel)
            idx = np.logical_and(pressure > el_p, pressure < lcl_p)
        else:
            idx = np.arange(0, len(pressure))
        return self.shade_area(pressure[idx], t_parcel[idx], t[idx], which='negative',
                               **kwargs)


@exporter.export
class Hodograph:
    r"""Make a hodograph of wind data.

    Plots the u and v components of the wind along the x and y axes, respectively.

    This class simplifies the process of creating a hodograph using matplotlib.
    It provides helpers for creating a circular grid and for plotting the wind as a line
    colored by another value (such as wind speed).

    Attributes
    ----------
    ax : `matplotlib.axes.Axes`
        The underlying Axes instance used for all plotting

    """

    def __init__(self, ax=None, component_range=80):
        r"""Create a Hodograph instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            The `Axes` instance used for plotting
        component_range : int
            The maximum range of the plot. Used to set plot bounds and control the maximum
            number of grid rings needed.

        """
        if ax is None:
            import matplotlib.pyplot as plt
            self.ax = plt.figure().add_subplot(1, 1, 1)
        else:
            self.ax = ax
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(-component_range, component_range)
        self.ax.set_ylim(-component_range, component_range)

        # == sqrt(2) * max_range, which is the distance at the corner
        self.max_range = 1.4142135 * component_range

    def add_grid(self, increment=10., **kwargs):
        r"""Add grid lines to hodograph.

        Creates lines for the x- and y-axes, as well as circles denoting wind speed values.

        Parameters
        ----------
        increment : int, optional
            The value increment between rings
        kwargs
            Other kwargs to control appearance of lines

        See Also
        --------
        :class:`matplotlib.patches.Circle`
        :meth:`matplotlib.axes.Axes.axhline`
        :meth:`matplotlib.axes.Axes.axvline`

        """
        # Some default arguments. Take those, and update with any
        # arguments passed in
        grid_args = {'color': 'grey', 'linestyle': 'dashed', 'zorder': Line2D.zorder - 0.001}
        if kwargs:
            grid_args.update(kwargs)

        # Take those args and make appropriate for a Circle
        circle_args = grid_args.copy()
        color = circle_args.pop('color', None)
        circle_args['edgecolor'] = color
        circle_args['fill'] = False

        self.rings = []
        for r in np.arange(increment, self.max_range, increment):
            c = Circle((0, 0), radius=r, **circle_args)
            self.ax.add_patch(c)
            self.rings.append(c)

        # Add lines for x=0 and y=0
        self.yaxis = self.ax.axvline(0, **grid_args)
        self.xaxis = self.ax.axhline(0, **grid_args)

    @staticmethod
    def _form_line_args(kwargs):
        """Simplify taking the default line style and extending with kwargs."""
        def_args = {'linewidth': 3}
        def_args.update(kwargs)
        return def_args

    def plot(self, u, v, **kwargs):
        r"""Plot u, v data.

        Plots the wind data on the hodograph.

        Parameters
        ----------
        u : array-like
            u-component of wind
        v : array-like
            v-component of wind
        kwargs
            Other keyword arguments to pass to :meth:`matplotlib.axes.Axes.plot`

        Returns
        -------
        list[matplotlib.lines.Line2D]
            lines plotted

        See Also
        --------
        :meth:`Hodograph.plot_colormapped`

        """
        line_args = self._form_line_args(kwargs)
        u, v = _delete_masked_points(u, v)
        return self.ax.plot(u, v, **line_args)

    def wind_vectors(self, u, v, **kwargs):
        r"""Plot u, v data as wind vectors.

        Plot the wind data as vectors for each level, beginning at the origin.

        Parameters
        ----------
        u : array-like
            u-component of wind
        v : array-like
            v-component of wind
        kwargs
            Other keyword arguments to pass to :meth:`matplotlib.axes.Axes.quiver`

        Returns
        -------
        matplotlib.quiver.Quiver
            arrows plotted

        """
        quiver_args = {'units': 'xy', 'scale': 1}
        quiver_args.update(**kwargs)
        center_position = np.zeros_like(u)
        return self.ax.quiver(center_position, center_position,
                              u, v, **quiver_args)

    def plot_colormapped(self, u, v, c, intervals=None, colors=None, **kwargs):
        r"""Plot u, v data, with line colored based on a third set of data.

        Plots the wind data on the hodograph, but with a colormapped line. Takes a third
        variable besides the winds (e.g. heights or pressure levels) and either a colormap to
        color it with or a series of contour intervals and colors to create a colormap and
        norm to control colormapping. The intervals must always be in increasing
        order. For using custom contour intervals with height data, the function will
        automatically interpolate to the contour intervals from the height and wind data,
        as well as convert the input contour intervals from height AGL to MSL to work with the
        provided heights.

        Parameters
        ----------
        u : array-like
            u-component of wind
        v : array-like
            v-component of wind
        c : array-like
            data to use for colormapping (e.g. heights, pressure, wind speed)
        intervals: array-like, optional
            Array of intervals for c to use in coloring the hodograph.
        colors: list, optional
            Array of strings representing colors for the hodograph segments.
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.LineCollection`

        Returns
        -------
        matplotlib.collections.LineCollection
            instance created

        See Also
        --------
        :meth:`Hodograph.plot`

        """
        u, v, c = _delete_masked_points(u, v, c)

        # Plotting a color segmented hodograph
        if colors:
            cmap = mcolors.ListedColormap(colors)
            # If we are segmenting by height (a length), interpolate the contour intervals
            if intervals.check('[length]'):

                # Find any intervals not in the data and interpolate them
                heights_min = np.nanmin(c)
                heights_max = np.nanmax(c)
                interpolation_heights = np.array([bound.m for bound in intervals
                                                  if bound not in c
                                                  and heights_min <= bound <= heights_max])
                interpolation_heights = units.Quantity(np.sort(interpolation_heights),
                                                       intervals.units)
                interpolated_u, interpolated_v = interpolate_1d(interpolation_heights, c, u, v)

                # Combine the interpolated data with the actual data
                c = concatenate([c, interpolation_heights])
                u = concatenate([u, interpolated_u])
                v = concatenate([v, interpolated_v])
                sort_inds = np.argsort(c)
                c = c[sort_inds]
                u = u[sort_inds]
                v = v[sort_inds]

                # Unit conversion required for coloring of bounds/data in dissimilar units
                # to work properly.
                c = c.to_base_units()  # TODO: This shouldn't be required!
                intervals = intervals.to_base_units()

            norm = mcolors.BoundaryNorm(intervals.magnitude, cmap.N)
            cmap.set_over('none')
            cmap.set_under('none')
            kwargs['cmap'] = cmap
            kwargs['norm'] = norm
            line_args = self._form_line_args(kwargs)

        # Plotting a continuously colored line
        else:
            line_args = self._form_line_args(kwargs)

        # Do the plotting
        lc = colored_line(u, v, c, **line_args)
        self.ax.add_collection(lc)
        return lc
