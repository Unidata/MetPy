__all__ = ['meteogram', 'station_plot']

from datetime import timedelta
from pytz import UTC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.dates import (DateFormatter, HourLocator, AutoDateLocator,
    date2num)
from metpy.cbook import iterable, get_title

#Default units for certain variables
default_units = {'temperature':'C', 'dewpoint':'C', 'relative humidity':'%',
    'pressure':'mb', 'wind speed':'m/s', 'solar radiation':'$W/m^2$',
    'rainfall':'mm', 'wind gusts':'m/s'}

def _rescale_yaxis(ax, bounds):
    # Manually tweak the limits here to ignore the low bottom set
    # for fill_between
    XY = np.array([[0]*len(bounds), bounds]).T
    ax.ignore_existing_data_limits = True
    ax.update_datalim(XY, updatex=False, updatey=True)
    ax.autoscale_view()

#TODO: REWRITE AS CLASS
def meteogram(data, fig=None, num_panels=3, time_range=None, ticker=None,
    layout=None, styles=None, limits=None, units=None, tz=UTC):
    '''
    Plots a meteogram (collection of time series) for a data set. This
    is broken down into a series of panels (defaults to 3), each of which
    can plot multiple variables, with sensible defaults, but can also be
    specified using *layout*.

    *data* : numpy record array
        A numpy record array containing time series for individual variables
        in each field.

    *fig* : :class:`matplotlib.figure.Figure` instance or None.
        A matplotlib Figure on which to draw.  If None, a new figure
        will be created.

    *num_panels* : int
        The number of panels to use in the plot.

    *time_range* : sequence, datetime.timedetla, or *None*
        If a sequence, the starting and ending times for the x-axis.  If
        a :class:`datetime.timedelta` object, it represents the time span
        to plot, which will end with the last data point.  It defaults to
        the last 24 hours of data.

    *ticker* : :class:`matplotlib.dates.DateLocator`
        An instance of a :class:`matplotlib.dates.DateLocator` that controls
        where the ticks will be located.  The default is
        :class:`matplotlib.dates.AutoDateLocator`.

    *layout* : dictionary
        A dictionary that maps panel numbers to lists of variables.
        If a panel is not found in the dictionary, a default (up to panel 5)
        will be used.  *None* can be included in the list to denote that
        :func:`pyplot.twinx` should be called, and the remaining variables
        plotted.

    *styles* : dictionary
        A dictionary that maps variable names to dictionary of matplotlib
        style arguments.  Also, the keyword `fill` can be included, to
        indicate that a filled plot should be used.  Any variable not
        specified will use a default (if available).

    *limits* : dictionary
        A dictionary that maps variable names to plot limits.  These limits
        are given by tuples with at least two items, which specify the
        start and end limits.  Either can be *None* which implies using the
        automatically determined value.  Optional third and fourth values
        can be given in the tuple, which is a list of tick values and labels,
        respectively.

    *units* : dictionary
        A dictionary that maps variable names to unit strings for axis labels.

    *tz* : datetime.tzinfo instance
        A :class:`datetime.tzinfo instance specifying the timezone to use
        for plotting the x-axis.  See the docs for :module:`datetime` and
        :module:`pytz` for how to construct and use these objects.  The
        default is UTC.

    Returns : list
        A list of the axes objects that were created.
    '''

    if fig is None:
        fig = plt.figure()

    #Get the time variable
    time = data['datetime']

    #Process time_range.
    if time_range is None:
        time_range = timedelta(hours=24)
        if ticker is None:
            ticker = HourLocator(byhour=np.arange(0, 25, 3), tz=tz)

    #Process ticker
    if ticker is None:
        ticker = AutoDateLocator(tz=tz)

    if not iterable(time_range):
        end = time[-1]
        start = end - time_range
        time_range = (start, end)

    #List of variables in each panel.  None denotes that at that point, twinx
    #should be called and the remaining variables plotted on the other axis
    default_layout = {
        0:['temperature', 'dewpoint'],
        1:['wind gusts', 'wind speed', None, 'wind direction'],
        2:['pressure'],
        3:['rainfall'],
        4:['solar radiation']}

    if layout is not None:
        default_layout.update(layout)
    layout = default_layout

    #Default styles for each variable
    default_styles = {
        'relative humidity':dict(color='#255425', linestyle='--'),
        'dewpoint':dict(facecolor='#265425', edgecolor='None', fill=True),
        'temperature':dict(facecolor='#C14F53', edgecolor='None', fill=True),
        'pressure':dict(facecolor='#895125', edgecolor='None', fill=True),
        'dewpoint':dict(facecolor='#265425', edgecolor='None', fill=True),
        'wind speed':dict(facecolor='#1C2386', edgecolor='None', fill=True),
        'wind gusts':dict(facecolor='#8388FC', edgecolor='None', fill=True),
        'wind direction':dict(markeredgecolor='#A9A64B', marker='D',
            linestyle='', markerfacecolor='None', markeredgewidth=1,
            markersize=3),
        'rainfall':dict(facecolor='#37CD37', edgecolor='None', fill=True),
        'solar radiation':dict(facecolor='#FF8529', edgecolor='None',
            fill=True),
        'windchill':dict(color='#8388FC', linewidth=1.5),
        'heat index':dict(color='#671A5C')}

    if styles is not None:
        default_styles.update(styles)
    styles = default_styles

    #Default data limits
    default_limits = {
        'wind direction':(0, 360, np.arange(0,400,45,),
            ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']),
        'wind speed':(0, None),
        'wind gusts':(0, None),
        'rainfall':(0, None),
        'solar radiation':(0, 1000, np.arange(0,1050,200))
        }
    if limits is not None:
        default_limits.update(limits)
    limits = default_limits

    #Set data units
    def_units = default_units.copy()
    if units is not None:
        def_units.update(units)
    units = def_units

    #Get the station name
    site = data['site'][0]

    #Get strings for the start and end times
    start = time_range[0].strftime('%H%MZ %d %b %Y')
    end = time_range[1].strftime('%H%MZ %d %b %Y')

    axes = []
    for panel in range(num_panels):
        if panel > 0:
            ax = fig.add_subplot(num_panels, 1, panel+1, sharex=ax)
        else:
            ax = fig.add_subplot(num_panels, 1, panel+1)
            ax.set_title('%s\n%s to %s' % (site, start, end))

        panel_twinned = False

        var_min = []
        var_max = []
        for varname in layout[panel]:
            if varname is None:
                _rescale_yaxis(ax, var_min + var_max)
                ax = ax.twinx()
                panel_twinned = True
                var_min = []
                var_max = []
                continue

            # Get the linestyle for this variable
            style = styles.get(varname, dict())

            #Get the variable from the data and plot
            var = data[varname]

            #Set special limits if necessary
            lims = limits.get(varname, (None, None))

            #Store the max and min for auto scaling
            var_max.append(var.max())
            var_min.append(var.min())

            if style.pop('fill', False):
                #Plot the filled area.  Need latest Matplotlib for date support
                #with fill_betweeen
                lower = -500 if lims[0] is None else lims[0]
                ax.fill_between(time, lower, var, where=~var.mask, **style)
                _rescale_yaxis(ax, var_min + var_max)
            else:
                ax.plot(time, var, **style)

            #If then length > 2, then we have ticks and (maybe) labels
            if len(lims) > 2:
                other = lims[2:]
                lims = lims[:2]
                #Separate out the ticks and perhaps labels
                if len(other) == 1:
                    ax.set_yticks(other[0])
                else:
                    ticks,labels = other
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(labels)
            ax.set_ylim(*lims)

            # Set the label to the title-cased nice-version from the
            # field info with units, if given.
            if varname in units and units[varname]:
                unit_str = ' (%s)' % units[varname]
                if '^' in unit_str:
                    unit_str = '$' + unit_str + '$'
            else:
                unit_str = ''

            descr = get_title(data, varname)
            ax.set_ylabel(descr.title() + unit_str)

        ax.xaxis.set_major_locator(ticker)
        ax.xaxis.set_major_formatter(DateFormatter('%H', tz))
        if not panel_twinned:
            ax.yaxis.set_ticks_position('both')
            for tick in ax.yaxis.get_major_ticks():
                tick.label2On = True
        axes.append(ax)
    ax.set_xlabel('Hour (%s)' % tz.tzname(time[0]))
    ax.set_xlim(*time_range)

    return axes

from matplotlib.artist import Artist
from matplotlib.cbook import is_string_like
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
class TextCollection(Artist):
    def __init__(self,
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 linespacing=None,
                 **kwargs
                 ):

        Artist.__init__(self)
        if color is None:
            colors= rcParams['text.color']

        if fontproperties is None:
            fontproperties = FontProperties()
        elif is_string_like(fontproperties):
            fontproperties = FontProperties(fontproperties)

        self._animated = False
#        if is_string_like(text):
#            text = [text]

        self._textobjs = [Text(x[ind], y[ind], text[ind], color,
            verticalalignment, horizontalalignment, multialignment,
            fontproperties, rotation, linespacing, **kwargs)
            for ind in xrange(len(x))]

        self.update(kwargs)

    def draw(self, renderer):
        for t in self._textobjs:
            t.draw(renderer)

    def set_figure(self, fig):
        for t in self._textobjs:
            t.set_figure(fig)

    def is_transform_set(self):
        return all(t.is_transform_set() for t in self._textobjs)

    def get_transform(self):
        return self._textobjs[0].get_transform()

    def set_transform(self, trans):
        for t in self._textobjs:
            t.set_transform(trans)

    def set_clip_path(self, path):
        for t in self._textobjs:
            t.set_clip_path(path)

    def set_axes(self, ax):
        for t in self._textobjs:
            t.set_axes(ax)

def text_plot(ax, x, y, data, format='%.0f', loc=None, **kw):
    from matplotlib.cbook import delete_masked_points
    from matplotlib import transforms

    # Default to centered on point
    if loc is not None:
        x0,y0 = loc
        trans = ax.transData + transforms.Affine2D().translate(x0, y0)
    else:
        trans = ax.transData

    # Handle both callables and strings for format
    if is_string_like(format):
        formatter = lambda s: format % s
    else:
        formatter = format

    # Handle masked arrays
    x,y,data = delete_masked_points(x, y, data)

    # If there is nothing left after deleting the masked points, return None
    if not data.any():
        return None

    # Make the TextCollection object
    texts = [formatter(d) for d in data]
    text_obj = TextCollection(x, y, texts, horizontalalignment='center',
        verticalalignment='center', clip_on=True, transform=trans, **kw)

    # Add it to the axes
    ax.add_artist(text_obj)

    # Update plot range
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    w = maxx - minx
    h = maxy - miny

    # the pad is a little hack to deal with the fact that we don't
    # want to transform all the symbols whose scales are in points
    # to data coords to get the exact bounding box for efficiency
    # reasons.  It can be done right if this is deemed important
    padx, pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim(corners)
    ax.autoscale_view()
    return text_obj

#Maps specifiers to normalized offsets in x and y
direction_map = dict(N=(0,1), NE=(1,1), E=(1,0), SE=(1,-1), S=(0,-1),
    SW=(-1,-1), W=(-1,0), NW=(-1,1), C=(0,0))

def station_plot(data, ax=None, proj=None, layout=None, styles=None,
    formats=None, offset=10.):
    '''
    Makes a station plot of the variables in data.

    *data* : numpy record array
        A numpy record array containing time series for individual variables
        in each field.

    *ax* : :class:`matplotlib.axes.Axes` instance or None
        The matplotlib Axes object on which to draw the station plot.  If None,
        the current Axes object is used.

    *proj* : callable or None
        A function to use to convert geographic coordinates, such as
        a :class:`mpl_toolkits.basemap.Basemap` instance.  If None,
        the geographic coordinates are used, as is, without any projection.

    *layout* : dictionary
        A dictionary that maps locations to field names.  Valid locations are:
        ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'C'], where 'C' represents
        the center location, and all others represent cardinal directions
        relative to the center point.  The actual distance from the center
        is controlled by *offset*.  A location can be mapped to None to
        specify that nothing should be plotted in that location.  This is
        useful for overriding the default layout.

    *styles* : dictionary
        A dictionary that maps variable names to dictionary of matplotlib
        text style arguments. Any variable not specified will use a
        default (if available).

    *formats* : dictionary
        A dictionary that maps variable names to either a format string or
        a function that takes data and returns a string. The default is to
        use '%0.f' for each variable.

    *offset* : float
        The offset, in pixels, from the center point for the values being
        plotted.

    Returns : dictionary
        A dictionary mapping layout locations to the object that was
        created in making the plot.
    '''
    if ax is None:
        ax = plt.gca()

    #If we don't get a lat/lon converter, make it a do-nothing function
    if proj is None:
        proj = lambda x,y:(x,y)

    #Update the default layout with the passed in one
    default_layout=dict(NW='temperature', SW='dewpoint', C=('u', 'v'))
    if layout is not None:
        default_layout.update(layout)
    layout = default_layout

    default_styles = {
        'temperature':dict(color='red'),
        'dewpoint':dict(color='green'),
        'wind speed':dict(color='blue'),
        'pressure':dict(color='black'),
        'rainfall':dict(color='blue')}
    if styles is not None:
        default_styles.update(styles)
    styles = default_styles

    if formats is None:
        formats = {}

    #Convert coordinates
    x,y = proj(data['longitude'], data['latitude'])

    results = dict()
    for spot in layout:
        var = layout[spot]
        # This allows specifying a spot in the layout as None to override
        # the default and eliminate a variable from the plot.
        if var is None:
            continue
        if len(var) == 2:
            # plot barbs.
            u,v = [v for v in var]
            style = styles.get('barbs', {})
            out = ax.barbs(x, y, data[u], data[v], **style)
        else:
            var = var
            style = styles.get(var, {})
            loc = map(lambda x:offset * x, direction_map[spot])
            f = formats.get(var, '%.0f')
            out = text_plot(ax, x, y, data[var], f, loc=loc, **style)

        # Add the output of the plotting command to the dictionary for
        # returning objects to the caller
        results[spot] = out

    return results
