from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
from matplotlib.projections import register_projection
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from matplotlib.collections import LineCollection
from metpy.calc.basic import dry_lapse, moist_lapse, dewpoint, vapor_pressure
from scipy.constants import C2K, K2C
import numpy as np


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

    def _gen_axes_spines(self):
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


class SkewT(object):
    '''
    Creates SkewT - logP plots.

    Kwargs:

        rotation: number
        Controls the rotation of temperature relative to horizontal. Given
        in degrees counterclockwise from x-axis.
    '''
    def __init__(self, fig, rotation=30):
        self._fig = fig
        self.ax = fig.add_subplot(1, 1, 1, projection='skewx',
                                  rotation=rotation)
        self.ax.grid(True)

    def plot(self, p, T, *args, **kwargs):
        # Skew-T logP plotting
        self.ax.semilogy(T, p, *args, **kwargs)

        # Disables the log-formatting that comes with semilogy
        self.ax.yaxis.set_major_formatter(ScalarFormatter())
        self.ax.yaxis.set_major_locator(MultipleLocator(100))
        if not self.ax.yaxis_inverted():
            self.ax.invert_yaxis()

        # Try to make sane default temperature plotting
        self.ax.xaxis.set_major_locator(MultipleLocator(10))
        self.ax.set_xlim(-50, 50)

    def plot_barbs(self, p, u, v, xloc=1.0, **kwargs):
        # Assemble array of x-locations in axes space
        x = np.empty_like(p)
        x.fill(xloc)

        # Do barbs plot at this location
        self.ax.barbs(x, p, u, v,
                      transform=self.ax.get_yaxis_transform(which='tick2'),
                      clip_on=False, **kwargs)

    def plot_dry_adiabats(self, T0=None, P=None, **kwargs):
        # Determine set of starting temps if necessary
        if T0 is None:
            xmin, xmax = self.ax.get_xlim()
            T0 = np.arange(xmin, xmax + 1, 10)

        # Get pressure levels based on ylims if necessary
        if P is None:
            P = np.linspace(*self.ax.get_ylim())

        # Assemble into data for plotting
        T = K2C(dry_lapse(P, C2K(T0[:, np.newaxis])))
        linedata = [np.vstack((t, P)).T for t in T]

        # Add to plot
        kwargs.setdefault('colors', 'r')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.5)
        self.ax.add_collection(LineCollection(linedata, **kwargs))

    def plot_moist_adiabats(self, T0=None, P=None, **kwargs):
        # Determine set of starting temps if necessary
        if T0 is None:
            xmin, xmax = self.ax.get_xlim()
            T0 = np.concatenate((np.arange(xmin, 0, 10),
                                 np.arange(0, xmax + 1, 5)))

        # Get pressure levels based on ylims if necessary
        if P is None:
            P = np.linspace(*self.ax.get_ylim())

        # Assemble into data for plotting
        T = K2C(moist_lapse(P, C2K(T0[:, np.newaxis])))
        linedata = [np.vstack((t, P)).T for t in T]

        # Add to plot
        kwargs.setdefault('colors', 'b')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.5)
        self.ax.add_collection(LineCollection(linedata, **kwargs))

    def plot_mixing_lines(self, w=None, P=None, **kwargs):
        # Default mixing level values if necessary
        if w is None:
            w = np.array([0.0004, 0.001, 0.002, 0.004, 0.007, 0.01,
                          0.016, 0.024, 0.032]).reshape(-1, 1)

        # Set pressure range if necessary
        if P is None:
            P = np.linspace(600, max(self.ax.get_ylim()))

        # Assemble data for plotting
        Td = dewpoint(vapor_pressure(P, w))
        linedata = [np.vstack((t, P)).T for t in Td]

        # Add to plot
        kwargs.setdefault('colors', 'g')
        kwargs.setdefault('linestyles', 'dashed')
        kwargs.setdefault('alpha', 0.8)
        self.ax.add_collection(LineCollection(linedata, **kwargs))


__all__ = ['SkewT']
