#  Copyright (c) 2019 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Test the simplified plotting interface."""

from datetime import datetime, timedelta
from io import BytesIO
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import pandas as pd
import pytest
from traitlets import TraitError
import xarray as xr

from metpy.cbook import get_test_data
from metpy.io import GiniFile
from metpy.io.metar import parse_metar_file
from metpy.plots import (BarbPlot, ContourPlot, FilledContourPlot, ImagePlot, MapPanel,
                         PanelContainer, PlotObs)
# Fixtures to make sure we have the right backend
from metpy.testing import set_agg_backend  # noqa: F401, I202
from metpy.units import units


MPL_VERSION = matplotlib.__version__[:3]


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.005)
def test_declarative_image():
    """Test making an image plot."""
    data = xr.open_dataset(GiniFile(get_test_data('NHEM-MULTICOMP_1km_IR_20151208_2100.gini')))

    img = ImagePlot()
    img.data = data.metpy.parse_cf('IR')
    img.colormap = 'Greys_r'

    panel = MapPanel()
    panel.title = 'Test'
    panel.plots = [img]

    pc = PanelContainer()
    pc.panel = panel
    pc.draw()

    assert panel.ax.get_title() == 'Test'

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.256}.get(MPL_VERSION, 0.022))
def test_declarative_contour():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa
    contour.contours = 30
    contour.linewidth = 1
    contour.linecolor = 'red'

    panel = MapPanel()
    panel.area = 'us'
    panel.proj = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.fixture
def fix_is_closed_polygon(monkeypatch):
    """Fix matplotlib.contour._is_closed_polygons for tests.

    Needed because for Matplotlib<3.3, the internal matplotlib.contour._is_closed_polygon
    uses strict floating point equality. This causes the test below to yield different
    results for macOS vs. Linux/Windows.

    """
    monkeypatch.setattr(matplotlib.contour, '_is_closed_polygon',
                        lambda X: np.allclose(X[0], X[-1], rtol=1e-10, atol=1e-13),
                        raising=False)


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 5.477}.get(MPL_VERSION, 0.035))
def test_declarative_contour_options(fix_is_closed_polygon):
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa
    contour.contours = 30
    contour.linewidth = 1
    contour.linecolor = 'red'
    contour.linestyle = 'dashed'
    contour.clabels = True

    panel = MapPanel()
    panel.area = 'us'
    panel.proj = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 2.007}.get(MPL_VERSION, 0.035))
def test_declarative_contour_convert_units(fix_is_closed_polygon):
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa
    contour.contours = 30
    contour.linewidth = 1
    contour.linecolor = 'red'
    contour.linestyle = 'dashed'
    contour.clabels = True
    contour.plot_units = 'degC'

    panel = MapPanel()
    panel.area = 'us'
    panel.proj = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.016)
def test_declarative_events():
    """Test that resetting traitlets properly propagates."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 850 * units.hPa
    contour.contours = 30
    contour.linewidth = 1
    contour.linecolor = 'red'

    img = ImagePlot()
    img.data = data
    img.field = 'v_wind'
    img.level = 700 * units.hPa
    img.colormap = 'hot'
    img.image_range = (3000, 5000)

    panel = MapPanel()
    panel.area = 'us'
    panel.proj = 'lcc'
    panel.layers = ['coastline', 'borders', 'states']
    panel.plots = [contour, img]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    # Update some properties to make sure it regenerates the figure
    contour.linewidth = 2
    contour.linecolor = 'green'
    contour.level = 700 * units.hPa
    contour.field = 'Specific_humidity'
    img.field = 'Geopotential_height'
    img.colormap = 'plasma'
    img.colorbar = 'horizontal'

    return pc.figure


def test_no_field_error():
    """Make sure we get a useful error when the field is not set."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa

    with pytest.raises(ValueError):
        contour.draw()


def test_no_field_error_barbs():
    """Make sure we get a useful error when the field is not set."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    barbs = BarbPlot()
    barbs.data = data
    barbs.level = 700 * units.hPa

    with pytest.raises(TraitError):
        barbs.draw()


def test_projection_object():
    """Test that we can pass a custom map projection."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.projection = ccrs.Mercator()
    panel.layers = [cfeature.LAKES]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.016)
def test_colorfill():
    """Test that we can use ContourFillPlot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = FilledContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'
    contour.colormap = 'coolwarm'
    contour.colorbar = 'vertical'

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.layers = [cfeature.STATES]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (12, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.016)
def test_colorfill_horiz_colorbar():
    """Test that we can use ContourFillPlot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = FilledContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'
    contour.colormap = 'coolwarm'
    contour.colorbar = 'horizontal'

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.layers = [cfeature.STATES]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (8, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.355}.get(MPL_VERSION, 0.016))
def test_colorfill_no_colorbar():
    """Test that we can use ContourFillPlot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = FilledContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'
    contour.colormap = 'coolwarm'
    contour.colorbar = None

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.layers = [cfeature.STATES]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (8, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=1.23)
def test_global():
    """Test that we can set global extent."""
    data = xr.open_dataset(GiniFile(get_test_data('NHEM-MULTICOMP_1km_IR_20151208_2100.gini')))

    img = ImagePlot()
    img.data = data
    img.field = 'IR'
    img.colorbar = None

    panel = MapPanel()
    panel.area = 'global'
    panel.plots = [img]

    pc = PanelContainer()
    pc.panel = panel
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.xfail(xr.__version__ < '0.11.0', reason='Does not work with older xarray.')
def test_latlon():
    """Test our handling of lat/lon information."""
    data = xr.open_dataset(get_test_data('irma_gfs_example.nc', as_file_obj=False))

    img = ImagePlot()
    img.data = data
    img.field = 'Temperature_isobaric'
    img.level = 500 * units.hPa
    img.time = datetime(2017, 9, 5, 15, 0, 0)
    img.colorbar = None

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height_isobaric'
    contour.level = img.level
    contour.time = img.time

    panel = MapPanel()
    panel.projection = 'lcc'
    panel.area = 'us'
    panel.plots = [img, contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.418}.get(MPL_VERSION, 0.37))
def test_declarative_barb_options():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    barb = BarbPlot()
    barb.data = data
    barb.level = 300 * units.hPa
    barb.field = ['u_wind', 'v_wind']
    barb.skip = (10, 10)
    barb.color = 'blue'
    barb.pivot = 'tip'
    barb.barblength = 6.5

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'data'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [barb]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.819}.get(MPL_VERSION, 0.612))
def test_declarative_barb_earth_relative():
    """Test making a contour plot."""
    import numpy as np
    data = xr.open_dataset(get_test_data('NAM_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height_isobaric'
    contour.level = 300 * units.hPa
    contour.linecolor = 'red'
    contour.linestyle = '-'
    contour.linewidth = 2
    contour.contours = np.arange(0, 20000, 120).tolist()

    barb = BarbPlot()
    barb.data = data
    barb.level = 300 * units.hPa
    barb.time = datetime(2016, 10, 31, 12)
    barb.field = ['u-component_of_wind_isobaric', 'v-component_of_wind_isobaric']
    barb.skip = (5, 5)
    barb.color = 'black'
    barb.barblength = 6.5
    barb.earth_relative = False

    panel = MapPanel()
    panel.area = (-124, -72, 20, 53)
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour, barb]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.612)
def test_declarative_gridded_scale():
    """Test making a contour plot."""
    import numpy as np
    data = xr.open_dataset(get_test_data('NAM_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height_isobaric'
    contour.level = 300 * units.hPa
    contour.linewidth = 2
    contour.contours = np.arange(0, 2000, 12).tolist()
    contour.scale = 1e-1
    contour.clabels = True

    panel = MapPanel()
    panel.area = (-124, -72, 20, 53)
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.607)
def test_declarative_barb_gfs():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('GFS_test.nc', as_file_obj=False))

    barb = BarbPlot()
    barb.data = data
    barb.level = 300 * units.hPa
    barb.field = ['u-component_of_wind_isobaric', 'v-component_of_wind_isobaric']
    barb.skip = (2, 2)
    barb.earth_relative = False

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'data'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [barb]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    barb.level = 700 * units.hPa

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.466)
def test_declarative_barb_gfs_knots():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('GFS_test.nc', as_file_obj=False))

    barb = BarbPlot()
    barb.data = data
    barb.level = 300 * units.hPa
    barb.field = ['u-component_of_wind_isobaric', 'v-component_of_wind_isobaric']
    barb.skip = (3, 3)
    barb.earth_relative = False
    barb.plot_units = 'knot'

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'data'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [barb]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.407}.get(MPL_VERSION, 0.022))
def test_declarative_sfc_obs():
    """Test making a surface observation plot."""
    data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                       infer_datetime_format=True, parse_dates=['valid'])

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 12, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['tmpf']
    obs.colors = ['black']

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = ccrs.PlateCarree()
    panel.area = 'in'
    panel.layers = ['states']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 8.09}.get(MPL_VERSION, 0.022))
def test_declarative_sfc_text():
    """Test making a surface observation plot with text."""
    data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                       infer_datetime_format=True, parse_dates=['valid'])

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 12, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['station']
    obs.colors = ['black']
    obs.formats = ['text']

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = 'lcc'
    panel.area = 'in'
    panel.layers = ['states']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.407}.get(MPL_VERSION, 0.022))
def test_declarative_sfc_obs_changes():
    """Test making a surface observation plot, changing the field."""
    data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                       infer_datetime_format=True, parse_dates=['valid'])

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 12, 12)
    obs.level = None
    obs.fields = ['tmpf']
    obs.colors = ['black']
    obs.time_window = timedelta(minutes=15)

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = ccrs.PlateCarree()
    panel.area = 'in'
    panel.layers = ['states']
    panel.plots = [obs]
    panel.title = f'Surface Observations for {obs.time}'

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    pc.draw()

    obs.fields = ['dwpf']
    obs.colors = ['green']

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.378}.get(MPL_VERSION, 0.00586))
def test_declarative_colored_barbs():
    """Test making a surface plot with a colored barb (gh-1274)."""
    data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                       infer_datetime_format=True, parse_dates=['valid'])

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 12, 13)
    obs.level = None
    obs.vector_field = ('uwind', 'vwind')
    obs.vector_field_color = 'red'
    obs.reduce_points = .5

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = ccrs.PlateCarree()
    panel.area = 'NE'
    panel.layers = ['states']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'3.1': 9.771,
                                          '2.1': 9.785}.get(MPL_VERSION, 0.00651))
def test_declarative_sfc_obs_full():
    """Test making a full surface observation plot."""
    data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                       infer_datetime_format=True, parse_dates=['valid'])

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 12, 13)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['tmpf', 'dwpf', 'emsl', 'cloud_cover', 'wxsym']
    obs.locations = ['NW', 'SW', 'NE', 'C', 'W']
    obs.colors = ['red', 'green', 'black', 'black', 'blue']
    obs.formats = [None, None, lambda v: format(10 * v, '.0f')[-3:], 'sky_cover',
                   'current_weather']
    obs.vector_field = ('uwind', 'vwind')
    obs.reduce_points = 1

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.area = (-124, -72, 20, 53)
    panel.area = 'il'
    panel.projection = ccrs.PlateCarree()
    panel.layers = ['coastline', 'borders', 'states']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.08)
def test_declarative_upa_obs():
    """Test making a full upperair observation plot."""
    data = pd.read_csv(get_test_data('UPA_obs.csv', as_file_obj=False))

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 14, 0)
    obs.level = 500 * units.hPa
    obs.fields = ['temperature', 'dewpoint', 'height']
    obs.locations = ['NW', 'SW', 'NE']
    obs.formats = [None, None, lambda v: format(v, '.0f')[:3]]
    obs.vector_field = ('u_wind', 'v_wind')
    obs.vector_field_length = 7
    obs.reduce_points = 0

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.area = (-124, -72, 20, 53)
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'states', 'land']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (15, 10)
    pc.panels = [panel]

    pc.draw()

    obs.level = 300 * units.hPa

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.08)
def test_declarative_upa_obs_convert_barb_units():
    """Test making a full upperair observation plot."""
    data = pd.read_csv(get_test_data('UPA_obs.csv', as_file_obj=False))
    data.units = ''
    data.units = {'pressure': 'hPa', 'height': 'meters', 'temperature': 'degC',
                  'dewpoint': 'degC', 'direction': 'degrees', 'speed': 'knots',
                  'station': None, 'time': None, 'u_wind': 'knots', 'v_wind': 'knots',
                  'latitude': 'degrees', 'longitude': 'degrees'}

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 14, 0)
    obs.level = 500 * units.hPa
    obs.fields = ['temperature', 'dewpoint', 'height']
    obs.locations = ['NW', 'SW', 'NE']
    obs.formats = [None, None, lambda v: format(v, '.0f')[:3]]
    obs.vector_field = ('u_wind', 'v_wind')
    obs.vector_field_length = 7
    obs.vector_plot_units = 'm/s'
    obs.reduce_points = 0

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.area = (-124, -72, 20, 53)
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'states', 'land']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (15, 10)
    pc.panels = [panel]

    pc.draw()

    obs.level = 300 * units.hPa

    return pc.figure


def test_attribute_error_time():
    """Make sure we get a useful error when the time variable is not found."""
    data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                       infer_datetime_format=True, parse_dates=['valid'])
    data.rename(columns={'valid': 'vtime'}, inplace=True)

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 12, 12)
    obs.level = None
    obs.fields = ['tmpf']
    obs.time_window = timedelta(minutes=15)

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = ccrs.PlateCarree()
    panel.area = 'in'
    panel.layers = ['states']
    panel.plots = [obs]
    panel.title = f'Surface Observations for {obs.time}'

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    with pytest.raises(AttributeError):
        pc.draw()


def test_attribute_error_station():
    """Make sure we get a useful error when the station variable is not found."""
    data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                       infer_datetime_format=True, parse_dates=['valid'])
    data.rename(columns={'station': 'location'}, inplace=True)

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(1993, 3, 12, 12)
    obs.level = None
    obs.fields = ['tmpf']
    obs.time_window = timedelta(minutes=15)

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = ccrs.PlateCarree()
    panel.area = 'in'
    panel.layers = ['states']
    panel.plots = [obs]
    panel.title = f'Surface Observations for {obs.time}'

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    with pytest.raises(AttributeError):
        pc.draw()


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.407}.get(MPL_VERSION, 0.022))
def test_declarative_sfc_obs_change_units():
    """Test making a surface observation plot."""
    data = parse_metar_file(get_test_data('metar_20190701_1200.txt', as_file_obj=False),
                            year=2019, month=7)

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(2019, 7, 1, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['air_temperature']
    obs.color = ['black']
    obs.plot_units = ['degF']

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = ccrs.PlateCarree()
    panel.area = 'in'
    panel.layers = ['states']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance={'2.1': 0.09}.get(MPL_VERSION, 0.022))
def test_declarative_multiple_sfc_obs_change_units():
    """Test making a surface observation plot."""
    data = parse_metar_file(get_test_data('metar_20190701_1200.txt', as_file_obj=False),
                            year=2019, month=7)

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(2019, 7, 1, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['air_temperature', 'dew_point_temperature', 'air_pressure_at_sea_level']
    obs.locations = ['NW', 'W', 'NE']
    obs.colors = ['red', 'green', 'black']
    obs.reduce_points = 0.75
    obs.plot_units = ['degF', 'degF', None]

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = ccrs.PlateCarree()
    panel.area = 'in'
    panel.layers = ['states']
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (12, 12)
    pc.panels = [panel]

    pc.draw()

    return pc.figure


def test_save():
    """Test that our saving function works."""
    pc = PanelContainer()
    fobj = BytesIO()
    pc.save(fobj, format='png')

    fobj.seek(0)

    # Test that our file object had something written to it.
    assert fobj.read()


def test_show():
    """Test that show works properly."""
    pc = PanelContainer()

    # Matplotlib warns when using show with Agg
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        pc.show()


def test_panel():
    """Test the functionality of the panel property."""
    panel = MapPanel()

    pc = PanelContainer()
    pc.panels = [panel]

    assert pc.panel is panel

    pc.panel = panel
    assert pc.panel is panel
