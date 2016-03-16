# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from .util import colored_line
from ..calc import dry_lapse, moist_lapse, dewpoint, vapor_pressure
from ..units import units

from ..package_tools import Exporter

exporter = Exporter(globals())


# The sole purpose of this class is to look at the upper, lower, or total
# interval as appropriate and see what parts of the tick to draw, if any.
class SkewXTick(maxis.XTick):
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__name__)

        lower_interval = self.axes.xaxis.lower_interval
        upper_interval = self.axes.xaxis.upper_interval

        if self.gridOn and transforms.interval_contains(
                self.axes.xaxis.get_view_interval(), self.get_loc()):
            self.gridline.draw(renderer)

        if transforms.interval_contains(lower_interval, self.get_loc()):
            if self.tick1On:
                self.tick1line.draw(renderer)
            if self.label1On:
                self.label1.draw(renderer)

        if transforms.interval_contains(upper_interval, self.get_loc()):
            if self.tick2On:
                self.tick2line.draw(renderer)
            if self.label2On:
                self.label2.draw(renderer)

        renderer.close_group(self.__name__)


# This class exists to provide two separate sets of intervals to the tick,
# as well as create instances of the custom tick
class SkewXAxis(maxis.XAxis):
    def __init__(self, *args, **kwargs):
        maxis.XAxis.__init__(self, *args, **kwargs)
        self.upper_interval = 0.0, 1.0

    def _get_tick(self, major):
        return SkewXTick(self.axes, 0, '', major=major)

    @property
    def lower_interval(self):
        return self.axes.viewLim.intervalx

    def get_view_interval(self):
        return self.upper_interval[0], self.axes.viewLim.intervalx[1]


# This class exists to calculate the separate data range of the
# upper X-axis and draw the spine there. It also provides this range
# to the X-axis artist for ticking and gridlines
class SkewSpine(mspines.Spine):
    def _adjust_location(self):
        trans = self.axes.transDataToAxes.inverted()
        if self.spine_type == 'top':
            yloc = 1.0
        else:
            yloc = 0.0
        left = trans.transform_point((0.0, yloc))[0]
        right = trans.transform_point((1.0, yloc))[0]

        pts = self._path.vertices
        pts[0, 0] = left
        pts[1, 0] = right
        self.axis.upper_interval = (left, right)


# This class handles registration of the skew-xaxes as a projection as well
# as setting up the appropriate transformations. It also overrides standard
# spines and axes instances as appropriate.
class SkewXAxes(Axes):
    # The projection must specify a name.  This will be used be the
    # user to select the projection, i.e. ``subplot(111,
    # projection='skewx')``.
    name = 'skewx'

    def __init__(self, *args, **kwargs):
        # This needs to be popped and set before moving on
        self.rot = kwargs.pop('rotation', 30)
        Axes.__init__(self, *args, **kwargs)

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
        spines = {'top': SkewSpine.linear_spine(self, 'top'),
                  'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                  'left': mspines.Spine.linear_spine(self, 'left'),
                  'right': mspines.Spine.linear_spine(self, 'right')}
        return spines

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # Get the standard transform setup from the Axes base class
        Axes._set_lim_and_transforms(self)

        # Need to put the skew in the middle, after the scale and limits,
        # but before the transAxes. This way, the skew is done in Axes
        # coordinates thus performing the transform around the proper origin
        # We keep the pre-transAxes transform around for other users, like the
        # spines for finding bounds
        self.transDataToAxes = (self.transScale +
                                (self.transLimits +
                                 transforms.Affine2D().skew_deg(self.rot, 0)))

        # Create the full transform from Data to Pixels
        self.transData = self.transDataToAxes + self.transAxes

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.
        self._xaxis_transform = (transforms.blended_transform_factory(
            self.transScale + self.transLimits,
            transforms.IdentityTransform()) +
            transforms.Affine2D().skew_deg(self.rot, 0)) + self.transAxes

# Now register the projection with matplotlib so the user can select
# it.
register_projection(SkewXAxes)


@exporter.export
class SkewT(object):
    r'''Make Skew-T log-P plots of data

    This class simplifies the process of creating Skew-T log-P plots in
    using matplotlib. It handles requesting the appropriate skewed projection,
    and provides simplified wrappers to make it easy to plot data, add wind
    barbs, and add other lines to the plots (e.g. dry adiabats)

    Attributes
    ----------
    ax : `matplotlib.axes.Axes`
        The underlying Axes instance, which can be used for calling additional
        plot functions (e.g. `axvline`)
    '''

    def __init__(self, fig=None, rotation=30, subplot=(1, 1, 1)):
        r'''Creates SkewT - logP plots.

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
        '''

        if fig is None:
            import matplotlib.pyplot as plt
            figsize = plt.rcParams.get('figure.figsize', (7, 7))
            fig = plt.figure(figsize=figsize)
        self._fig = fig

        # Handle being passed a tuple for the subplot, or a GridSpec instance
        try:
            len(subplot)
        except TypeError:
            subplot = (subplot,)
        self.ax = fig.add_subplot(*subplot, projection='skewx', rotation=rotation)
        self.ax.grid(True)

    def plot(self, p, t, *args, **kwargs):
        r'''Plot data.

        Simple wrapper around plot so that pressure is the first (independent)
        input. This is essentially a wrapper around `semilogy`. It also
        sets some appropriate ticking and plot ranges.

        Parameters
        ----------
        p : array_like
            pressure values
        t : array_like
            temperature values, can also be used for things like dew point
        args
            Other positional arguments to pass to :func:`~matplotlib.pyplot.semilogy`
        kwargs
            Other keyword arguments to pass to :func:`~matplotlib.pyplot.semilogy`

        Returns
        -------
        list[matplotlib.lines.Line2D]
            lines plotted

        See Also
        --------
        :func:`matplotlib.pyplot.semilogy`
        '''

        # Skew-T logP plotting
        l = self.ax.semilogy(t, p, *args, **kwargs)

        # Disables the log-formatting that comes with semilogy
        self.ax.yaxis.set_major_formatter(ScalarFormatter())
        self.ax.yaxis.set_major_locator(MultipleLocator(100))
        if not self.ax.yaxis_inverted():
            self.ax.invert_yaxis()

        # Try to make sane default temperature plotting
        self.ax.xaxis.set_major_locator(MultipleLocator(10))
        self.ax.set_xlim(-50, 50)

        return l

    def plot_barbs(self, p, u, v, xloc=1.0, x_clip_radius=0.08, y_clip_radius=0.08, **kwargs):
        r'''Plot wind barbs.

        Adds wind barbs to the skew-T plot. This is a wrapper around the
        `barbs` command that adds to appropriate transform to place the
        barbs in a vertical line, located as a function of pressure.

        Parameters
        ----------
        p : array_like
            pressure values
        u : array_like
            U (East-West) component of wind
        v : array_like
            V (North-South) component of wind
        xloc : float, optional
            Position for the barbs, in normalized axes coordinates, where 0.0
            denotes far left and 1.0 denotes far right. Defaults to far right.
        x_clip_radius : float, optional
            Space, in normalized axes coordinates, to leave before clipping
            wind barbs in the x-direction. Defaults to 0.08.
        y_clip_radius : float, optional
            Space, in normalized axes coordinates, to leave above/below plot
            before clipping wind barbs in the y-direction. Defaults to 0.08.
        kwargs
            Other keyword arguments to pass to :func:`~matplotlib.pyplot.barbs`

        Returns
        -------
        matplotlib.quiver.Barbs
            instance created

        See Also
        --------
        :func:`matplotlib.pyplot.barbs`
        '''

        # Assemble array of x-locations in axes space
        x = np.empty_like(p)
        x.fill(xloc)

        # Do barbs plot at this location
        b = self.ax.barbs(x, p, u, v,
                          transform=self.ax.get_yaxis_transform(which='tick2'),
                          clip_on=True, **kwargs)

        # Override the default clip box, which is the axes rectangle, so we can have
        # barbs that extend outside.
        ax_bbox = transforms.Bbox([[xloc - x_clip_radius, -y_clip_radius],
                                   [xloc + x_clip_radius, 1.0 + y_clip_radius]])
        b.set_clip_box(transforms.TransformedBbox(ax_bbox, self.ax.transAxes))
        return b

    def plot_dry_adiabats(self, t0=None, p=None, **kwargs):
        r'''Plot dry adiabats.

        Adds dry adiabats (lines of constant potential temperature) to the
        plot. The default style of these lines is dashed red lines with an alpha
        value of 0.5. These can be overridden using keyword arguments.

        Parameters
        ----------
        t0 : array_like, optional
            Starting temperature values in Kelvin. If none are given, they will be
            generated using the current temperature range at the bottom of
            the plot.
        p : array_like, optional
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
        :func:`~metpy.calc.thermo.dry_lapse`
        :meth:`plot_moist_adiabats`
        :class:`matplotlib.collections.LineCollection`
        '''

        # Determine set of starting temps if necessary
        if t0 is None:
            xmin, xmax = self.ax.get_xlim()
            t0 = np.arange(xmin, xmax + 1, 10) * units.degC

        # Get pressure levels based on ylims if necessary
        if p is None:
            p = np.linspace(*self.ax.get_ylim()) * units.mbar

        # Assemble into data for plotting
        t = dry_lapse(p, t0[:, np.newaxis]).to(units.degC)
        linedata = [np.vstack((ti, p)).T for ti in t]

        # Add to plot
        kwargs.setdefault('colors', 'r')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.5)
        return self.ax.add_collection(LineCollection(linedata, **kwargs))

    def plot_moist_adiabats(self, t0=None, p=None, **kwargs):
        r'''Plot moist adiabats.

        Adds saturated pseudo-adiabats (lines of constant equivalent potential
        temperature) to the plot. The default style of these lines is dashed
        blue lines with an alpha value of 0.5. These can be overridden using
        keyword arguments.

        Parameters
        ----------
        t0 : array_like, optional
            Starting temperature values in Kelvin. If none are given, they will be
            generated using the current temperature range at the bottom of
            the plot.
        p : array_like, optional
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
        :func:`~metpy.calc.thermo.moist_lapse`
        :meth:`plot_dry_adiabats`
        :class:`matplotlib.collections.LineCollection`
        '''

        # Determine set of starting temps if necessary
        if t0 is None:
            xmin, xmax = self.ax.get_xlim()
            t0 = np.concatenate((np.arange(xmin, 0, 10),
                                 np.arange(0, xmax + 1, 5))) * units.degC

        # Get pressure levels based on ylims if necessary
        if p is None:
            p = np.linspace(*self.ax.get_ylim()) * units.mbar

        # Assemble into data for plotting
        t = moist_lapse(p, t0[:, np.newaxis]).to(units.degC)
        linedata = [np.vstack((ti, p)).T for ti in t]

        # Add to plot
        kwargs.setdefault('colors', 'b')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.5)
        return self.ax.add_collection(LineCollection(linedata, **kwargs))

    def plot_mixing_lines(self, w=None, p=None, **kwargs):
        r'''Plot lines of constant mixing ratio.

        Adds lines of constant mixing ratio (isohumes) to the
        plot. The default style of these lines is dashed green lines with an
        alpha value of 0.8. These can be overridden using keyword arguments.

        Parameters
        ----------
        w : array_like, optional
            Unitless mixing ratio values to plot. If none are given, default
            values are used.
        p : array_like, optional
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
        '''

        # Default mixing level values if necessary
        if w is None:
            w = np.array([0.0004, 0.001, 0.002, 0.004, 0.007, 0.01,
                          0.016, 0.024, 0.032]).reshape(-1, 1)

        # Set pressure range if necessary
        if p is None:
            p = np.linspace(600, max(self.ax.get_ylim())) * units.mbar

        # Assemble data for plotting
        td = dewpoint(vapor_pressure(p, w))
        linedata = [np.vstack((t, p)).T for t in td]

        # Add to plot
        kwargs.setdefault('colors', 'g')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.8)
        return self.ax.add_collection(LineCollection(linedata, **kwargs))


@exporter.export
class Hodograph(object):
    r'''Make a hodograph of wind data--plots the u and v components of the wind along the
    x and y axes, respectively.

    This class simplifies the process of creating a hodograph using matplotlib.
    It provides helpers for creating a circular grid and for plotting the wind as a line
    colored by another value (such as wind speed).

    Attributes
    ----------
    ax : `matplotlib.axes.Axes`
        The underlying Axes instance used for all plotting
    '''
    def __init__(self, ax=None, component_range=80):
        r'''Create a Hodograph instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            The `Axes` instance used for plotting
        component_range : value
            The maximum range of the plot. Used to set plot bounds and control the maximum
            number of grid rings needed.
        '''
        if ax is None:
            import matplotlib.pyplot as plt
            self.ax = plt.figure().add_subplot(1, 1, 1)
        else:
            self.ax = ax
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-component_range, component_range)
        ax.set_ylim(-component_range, component_range)

        # == sqrt(2) * max_range, which is the distance at the corner
        self.max_range = 1.4142135 * component_range

    def add_grid(self, increment=10., **kwargs):
        r'''Add grid lines to hodograph.

        Creates lines for the x- and y-axes, as well as circles denoting wind speed values.

        Parameters
        ----------
        increment : value, optional
            The value increment between rings
        kwargs
            Other kwargs to control appearance of lines

        See Also
        --------
        :class:`matplotlib.patches.Circle`
        :meth:`matplotlib.axes.Axes.axhline`
        :meth:`matplotlib.axes.Axes.axvline`
        '''
        # Some default arguments. Take those, and update with any
        # arguments passed in
        grid_args = dict(color='grey', linestyle='dashed')
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
        r'Simple helper to take default line style and extend with kwargs'
        def_args = dict(linewidth=3)
        def_args.update(kwargs)
        return def_args

    def plot(self, u, v, **kwargs):
        r'''Plot u, v data.

        Plots the wind data on the hodograph.

        Parameters
        ----------
        u : array_like
            u-component of wind
        v : array_like
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
        '''
        line_args = self._form_line_args(kwargs)
        return self.ax.plot(u, v, **line_args)

    def plot_colormapped(self, u, v, c, **kwargs):
        r'''Plot u, v data, with line colored based on a third set of data.

        Plots the wind data on the hodograph, but

        Simple wrapper around plot so that pressure is the first (independent)
        input. This is essentially a wrapper around `semilogy`. It also
        sets some appropriate ticking and plot ranges.

        Parameters
        ----------
        u : array_like
            u-component of wind
        v : array_like
            v-component of wind
        c : array_like
            data to use for colormapping
        kwargs
            Other keyword arguments to pass to :class:`matplotlib.collections.LineCollection`

        Returns
        -------
        matplotlib.collections.LineCollection
            instance created

        See Also
        --------
        :meth:`Hodograph.plot`
        '''
        line_args = self._form_line_args(kwargs)
        lc = colored_line(u, v, c, **line_args)
        self.ax.add_collection(lc)
        return lc
