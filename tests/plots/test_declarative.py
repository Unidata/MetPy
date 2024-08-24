#  Copyright (c) 2019 MetPy Developers.
#  Distributed under the terms of the BSD 3-Clause License.
#  SPDX-License-Identifier: BSD-3-Clause
"""Test the simplified plotting interface."""

from datetime import datetime, timedelta
from io import BytesIO
from unittest.mock import patch, PropertyMock
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from traitlets import TraitError
import xarray as xr

from metpy.calc import wind_speed
from metpy.cbook import get_test_data
from metpy.io import GiniFile
from metpy.io.metar import parse_metar_file
from metpy.plots import (ArrowPlot, BarbPlot, ContourPlot, FilledContourPlot, ImagePlot,
                         MapPanel, PanelContainer, PlotGeometry, PlotObs, RasterPlot)
from metpy.testing import needs_cartopy, version_check
from metpy.units import units


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.02)
@needs_cartopy
def test_declarative_image():
    """Test making an image plot."""
    data = xr.open_dataset(GiniFile(get_test_data('NHEM-MULTICOMP_1km_IR_20151208_2100.gini')))

    img = ImagePlot()
    img.data = data.metpy.parse_cf('IR')
    img.mpl_args = {'cmap': 'Greys_r'}

    panel = MapPanel()
    panel.title = 'Test'
    panel.plots = [img]

    pc = PanelContainer()
    pc.panel = panel
    pc.draw()

    assert panel.ax.get_title() == 'Test'

    return pc.figure


@needs_cartopy
def test_declarative_three_dims_error():
    """Test making an image plot with three dimensions."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    img = ImagePlot()
    img.data = data
    img.field = 'Temperature'
    img.colormap = 'coolwarm'

    panel = MapPanel()
    panel.plots = [img]

    pc = PanelContainer()
    pc.panel = panel

    with pytest.raises(ValueError, match='subset for plotting'):
        pc.draw()


@needs_cartopy
def test_declarative_four_dims_error():
    """Test making a contour plot with four dimensions."""
    data = xr.open_dataset(get_test_data('CAM_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'PN'
    contour.linecolor = 'black'
    contour.contours = list(range(0, 1200, 4))

    panel = MapPanel()
    panel.plots = [contour]
    panel.layout = (1, 1, 1)
    panel.layers = ['coastline', 'borders', 'states', 'land']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panels = [panel]

    with pytest.raises(ValueError, match='subset for plotting'):
        pc.draw()


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.163 if version_check('cartopy<0.23') else 0.09)
@needs_cartopy
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
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=False, tolerance=0.094)
@needs_cartopy
def test_declarative_titles():
    """Test making a contour plot with multiple titles."""
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
    panel.projection = 'lcc'
    panel.layers = ['coastline']
    panel.left_title = '700-hPa Temperature'
    panel.right_title = 'Valid at a time'
    panel.title = 'Plot of data'
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.159 if version_check('cartopy<0.23') else 0.066)
@needs_cartopy
def test_declarative_smooth_contour():
    """Test making a contour plot using smooth_contour."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa
    contour.contours = 30
    contour.linewidth = 1
    contour.linecolor = 'red'
    contour.smooth_contour = 5

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.155 if version_check('cartopy<0.23') else 0.006)
@needs_cartopy
def test_declarative_smooth_contour_calculation():
    """Test making a contour plot using smooth_contour."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))
    data = data.metpy.parse_cf().metpy.assign_latitude_longitude()

    data['wind_speed'] = wind_speed(data['u_wind'], data['v_wind'])

    contour = ContourPlot()
    contour.data = data
    contour.field = 'wind_speed'
    contour.level = 300 * units.hPa
    contour.contours = range(50, 211, 20)
    contour.linewidth = 1
    contour.linecolor = 'blue'
    contour.smooth_contour = 10
    contour.plot_units = 'kt'

    contour2 = ContourPlot()
    contour2.data = data
    contour2.field = 'Geopotential_height'
    contour2.level = 300 * units.hPa
    contour2.contours = range(0, 15000, 120)
    contour2.linewidth = 1
    contour2.linecolor = 'black'
    contour2.smooth_contour = 4

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour, contour2]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.142 if version_check('cartopy<0.23') else 0.0038)
@needs_cartopy
def test_declarative_smooth_contour_order():
    """Test making a contour plot using smooth_contour with tuple."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height'
    contour.level = 700 * units.hPa
    contour.contours = list(range(0, 4000, 30))
    contour.linewidth = 1
    contour.linecolor = 'black'
    contour.smooth_contour = (10, 4)

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.114 if version_check('cartopy<0.23') else 0.058)
@needs_cartopy
def test_declarative_figsize():
    """Test having an all float figsize."""
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
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (10.5, 10.5)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.104 if version_check('cartopy<0.23') else 0.033)
@needs_cartopy
def test_declarative_smooth_field():
    """Test the smoothing of the field with smooth_field trait."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height'
    contour.level = 700 * units.hPa
    contour.contours = list(range(0, 4000, 30))
    contour.linewidth = 1
    contour.linecolor = 'black'
    contour.smooth_field = 3

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (10.5, 10.5)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.828)
@needs_cartopy
def test_declarative_contour_cam():
    """Test making a contour plot with CAM data."""
    data = xr.open_dataset(get_test_data('CAM_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'PN'
    contour.time = datetime.strptime('2020-11-29 00:00', '%Y-%m-%d %H:%M')
    contour.level = 1000 * units.hPa
    contour.linecolor = 'black'
    contour.contours = list(range(0, 1200, 4))

    panel = MapPanel()
    panel.plots = [contour]
    panel.layout = (1, 1, 1)
    panel.layers = ['coastline', 'borders', 'states', 'land']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(
    remove_text=True,
    tolerance=3.71 if version_check('matplotlib<3.8') else 0.74)
@needs_cartopy
def test_declarative_contour_options():
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
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.152 if version_check('cartopy<0.23') else 0.009)
@needs_cartopy
def test_declarative_layers_plot_options():
    """Test declarative layer options of edgecolor and linewidth."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa
    contour.contours = 5
    contour.linewidth = 1
    contour.mpl_args = {'colors': 'grey'}

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'usstates', 'borders']
    panel.layers_edgecolor = ['blue', 'red', 'black']
    panel.layers_linewidth = [0.75, 0.75, 1]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.055 if version_check('cartopy<0.23') else 0.009)
@needs_cartopy
def test_declarative_additional_layers_plot_options():
    """Test additional declarative layer options of linestyle, zorder, and alpha."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa
    contour.contours = 5
    contour.linewidth = 1
    contour.linecolor = 'grey'

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'usstates', 'borders', 'lakes', 'rivers']
    panel.layers_edgecolor = ['blue', 'red', 'black', None, 'water']
    panel.layers_linewidth = [0.75, 0.75, 1, 1, 1]
    panel.layers_linestyle = ['solid', 'dotted', 'dashed', 'dotted']
    panel.layers_alpha = [1, .5, .75, 1]
    panel.layers_zorder = [1, 1, 1, -1, -1]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(
    remove_text=True,
    tolerance=(
        3.34 if version_check('cartopy<0.23') else
        2.74 if version_check('matplotlib<3.8') else 1.91))
@needs_cartopy
def test_declarative_contour_convert_units():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa
    contour.contours = 30
    contour.linewidth = 1
    contour.linecolor = 'red'
    contour.clabels = True
    contour.plot_units = 'degC'

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=2.731 if version_check('matplotlib<3.9') else 0.246)
@needs_cartopy
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
    panel.projection = 'lcc'
    panel.layers = []
    panel.plots = [contour, img]

    pc = PanelContainer()
    pc.size = (8, 8.0)
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.009)
@needs_cartopy
def test_declarative_raster_events():
    """Test that resetting traitlets properly propagates in RasterPlot()."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    raster = RasterPlot()
    raster.data = data
    raster.field = 'Temperature'
    raster.level = 700 * units.hPa
    raster.colormap = 'hot'

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.plots = [raster]

    pc = PanelContainer()
    pc.size = (8, 8.0)
    pc.panels = [panel]
    pc.draw()

    # Update some properties to make sure it regenerates the figure
    raster.level = 700 * units.hPa
    raster.field = 'Geopotential_height'
    raster.colormap = 'viridis'
    raster.colorbar = 'vertical'

    return pc.figure


def test_no_field_error():
    """Make sure we get a useful error when the field is not set."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa

    with pytest.raises(ValueError):
        contour.draw()


def test_ndim_error_scalar(cfeature):
    """Make sure we get a useful error when the field is not set."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = None

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.projection = 'lcc'
    panel.layers = [cfeature.LAKES]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel

    with pytest.raises(ValueError):
        pc.draw()

    plt.close(pc.figure)


def test_ndim_error_vector(cfeature):
    """Make sure we get a useful error when the field is not set."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    barbs = BarbPlot()
    barbs.data = data
    barbs.field = ['u_wind', 'v_wind']
    barbs.level = None

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.projection = 'lcc'
    panel.plots = [barbs]

    pc = PanelContainer()
    pc.panel = panel

    with pytest.raises(ValueError):
        pc.draw()

    plt.close(pc.figure)


def test_no_field_error_barbs():
    """Make sure we get a useful error when the field is not set."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    barbs = BarbPlot()
    barbs.data = data
    barbs.level = 700 * units.hPa

    with pytest.raises(TraitError):
        barbs.draw()


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.381)
def test_projection_object(ccrs, cfeature):
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.009)
@needs_cartopy
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
    panel.layers = []
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (12, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.238 if version_check('cartopy<0.23') else 0.004)
def test_colorfill_with_image_range(cfeature):
    """Test that we can use ContourFillPlot with image_range bounds."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = FilledContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'
    contour.colormap = 'coolwarm'
    contour.colorbar = None
    contour.image_range = (273.15, 350)

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.layers = [cfeature.STATES]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (8, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(
    remove_text=True,
    tolerance=0.238 if version_check('cartopy<0.23') else 0.004,
    filename='test_colorfill_with_image_range.png'
)
def test_colorfill_with_normalize_instance_image_range(cfeature):
    """Test that we can use ContourFillPlot with image_range bounds."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = FilledContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'
    contour.colormap = 'coolwarm'
    contour.colorbar = None
    contour.image_range = plt.Normalize(vmin=273.15, vmax=350)

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.layers = [cfeature.STATES]
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (8, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.02)
@needs_cartopy
def test_colorfill_horiz_colorbar():
    """Test that we can use ContourFillPlot with a horizontal colorbar."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = FilledContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'
    contour.colormap = 'coolwarm'
    contour.colorbar = 'horizontal'

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.layers = []
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (8, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.02)
def test_colorbar_kwargs(cfeature):
    """Test that we can use ContourFillPlot with specifying colorbar kwargs."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = FilledContourPlot()
    contour.data = data
    contour.level = 700 * units.hPa
    contour.field = 'Temperature'
    contour.colormap = 'coolwarm'
    contour.colorbar = {'orientation': 'horizontal', 'aspect': 60, 'pad': 0.05}

    panel = MapPanel()
    panel.area = (-110, -60, 25, 55)
    panel.layers = []
    panel.plots = [contour]

    pc = PanelContainer()
    pc.panel = panel
    pc.size = (8, 8)
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.370 if version_check('cartopy<0.23') else 0.005)
def test_colorfill_no_colorbar(cfeature):
    """Test that we can use ContourFillPlot with no colorbar."""
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
@needs_cartopy
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
@needs_cartopy
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.393)
@needs_cartopy
def test_declarative_barb_options():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    barb = BarbPlot()
    barb.data = data
    barb.level = 300 * units.hPa
    barb.field = ['u_wind', 'v_wind']
    barb.skip = (10, 10)
    barb.color = 'blue'
    barb.pivot = 'middle'
    barb.mpl_args = {'pivot': 'tip'}
    barb.barblength = 6.5

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'data'
    panel.layers = ['coastline', 'borders', 'usstates', 'land', 'ocean', 'lakes']
    panel.plots = [barb]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.37)
@needs_cartopy
def test_declarative_arrowplot():
    """Test making a arrow plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    arrows = ArrowPlot()
    arrows.data = data
    arrows.level = 300 * units.hPa
    arrows.field = ['u_wind', 'v_wind']
    arrows.skip = (10, 10)
    arrows.color = 'blue'
    arrows.pivot = 'tip'
    arrows.mpl_args = {'pivot': 'mid'}
    arrows.arrowscale = 1000

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'data'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [arrows]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.37)
@needs_cartopy
def test_declarative_arrowkey():
    """Test making a arrow plot with an arrow key."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    arrows = ArrowPlot()
    arrows.data = data
    arrows.level = 300 * units.hPa
    arrows.field = ['u_wind', 'v_wind']
    arrows.skip = (10, 10)
    arrows.color = 'red'
    arrows.pivot = 'mid'
    arrows.arrowscale = 1e3
    arrows.arrowkey = (100, None, 1.05, None, '100 kt')
    arrows.plot_units = 'kt'

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'data'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [arrows]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.37)
@needs_cartopy
def test_declarative_arrow_changes():
    """Test making a arrow plot with an arrow key."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    arrows = ArrowPlot()
    arrows.data = data
    arrows.level = 300 * units.hPa
    arrows.field = ['u_wind', 'v_wind']
    arrows.skip = (10, 10)
    arrows.color = 'red'
    arrows.pivot = 'mid'

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'data'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [arrows]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    arrows.color = 'green'
    arrows.arrowkey = (None, 0.9, 1.1, 'W', '100 m/s')

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.891)
@needs_cartopy
def test_declarative_barb_earth_relative():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('NAM_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height_isobaric'
    contour.level = 300 * units.hPa
    contour.linecolor = 'red'
    contour.linestyle = '-'
    contour.linewidth = 2
    contour.contours = range(0, 20000, 120)

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
@needs_cartopy
def test_declarative_overlay_projections():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('NAM_test.nc', as_file_obj=False))
    data2 = xr.open_dataset(get_test_data('GFS_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height_isobaric'
    contour.level = 300 * units.hPa
    contour.linecolor = 'red'
    contour.linestyle = '-'
    contour.linewidth = 2
    contour.contours = np.arange(0, 20000, 120).tolist()

    contour2 = ContourPlot()
    contour2.data = data2
    contour2.field = 'Geopotential_height_isobaric'
    contour2.level = 300 * units.hPa
    contour2.linecolor = 'blue'
    contour2.linestyle = '-'
    contour2.linewidth = 2
    contour2.contours = np.arange(0, 20000, 120).tolist()

    panel = MapPanel()
    panel.area = (-124, -72, 20, 53)
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour, contour2]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.133 if version_check('cartopy<0.23') else 0.0094)
@needs_cartopy
def test_declarative_gridded_scale():
    """Test making a contour plot."""
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
@needs_cartopy
def test_declarative_global_gfs():
    """Test making a global contour plot using GFS."""
    data = xr.open_dataset(get_test_data('GFS_global.nc', as_file_obj=False))

    cntr = ContourPlot()
    cntr.data = data
    cntr.time = datetime(2021, 1, 30, 12)
    cntr.field = 'Geopotential_height_isobaric'
    cntr.level = 300 * units.hPa
    cntr.contours = np.arange(0, 100000, 120).tolist()
    cntr.linecolor = 'darkblue'
    cntr.linewidth = 1

    panel = MapPanel()
    panel.area = [-180, 180, 10, 90]
    panel.projection = 'ps'
    panel.layers = ['coastline']
    panel.plots = [cntr]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=1.42)
@needs_cartopy
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.665)
@needs_cartopy
def test_declarative_barb_scale():
    """Test making a contour plot."""
    data = xr.open_dataset(get_test_data('GFS_test.nc', as_file_obj=False))

    barb = BarbPlot()
    barb.data = data
    barb.level = 300 * units.hPa
    barb.field = ['u-component_of_wind_isobaric', 'v-component_of_wind_isobaric']
    barb.skip = (3, 3)
    barb.earth_relative = False
    barb.scale = 2

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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.722)
@needs_cartopy
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


@pytest.fixture()
def sample_obs():
    """Generate sample observational data for testing."""
    return pd.DataFrame([('2020-08-05 12:00', 'KDEN', 1000, 1, 9),
                         ('2020-08-05 12:01', 'KOKC', 1000, 2, 10),
                         ('2020-08-05 12:00', 'KDEN', 500, 3, 11),
                         ('2020-08-05 12:01', 'KOKC', 500, 4, 12),
                         ('2020-08-06 13:00', 'KDEN', 1000, 5, 13),
                         ('2020-08-06 12:59', 'KOKC', 1000, 6, 14),
                         ('2020-08-06 13:00', 'KDEN', 500, 7, 15),
                         ('2020-08-06 12:59', 'KOKC', 500, 8, 16)],
                        columns=['time', 'stid', 'pressure', 'temperature', 'dewpoint'])


@pytest.fixture()
def pandas_sfc():
    """Open sample pandas data."""
    df = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False))
    df['valid'] = pd.to_datetime(df['valid'], format='%Y-%m-%d %H:%M:%S')
    return df


def test_plotobs_subset_default_nolevel(sample_obs):
    """Test PlotObs subsetting with minimal config."""
    obs = PlotObs()
    obs.data = sample_obs

    truth = pd.DataFrame([('2020-08-06 13:00', 'KDEN', 500, 7, 15),
                          ('2020-08-06 12:59', 'KOKC', 500, 8, 16)],
                         columns=['time', 'stid', 'pressure', 'temperature', 'dewpoint'],
                         index=[6, 7])
    pd.testing.assert_frame_equal(obs.obsdata, truth)


def test_plotobs_subset_level(sample_obs):
    """Test PlotObs subsetting based on level."""
    obs = PlotObs()
    obs.data = sample_obs
    obs.level = 1000 * units.hPa

    truth = pd.DataFrame([('2020-08-06 13:00', 'KDEN', 1000, 5, 13),
                          ('2020-08-06 12:59', 'KOKC', 1000, 6, 14)],
                         columns=['time', 'stid', 'pressure', 'temperature', 'dewpoint'],
                         index=[4, 5])
    pd.testing.assert_frame_equal(obs.obsdata, truth)


def test_plotobs_subset_level_no_units(sample_obs):
    """Test PlotObs subsetting based on unitless level."""
    obs = PlotObs()
    obs.data = sample_obs
    obs.level = 1000

    truth = pd.DataFrame([('2020-08-06 13:00', 'KDEN', 1000, 5, 13),
                          ('2020-08-06 12:59', 'KOKC', 1000, 6, 14)],
                         columns=['time', 'stid', 'pressure', 'temperature', 'dewpoint'],
                         index=[4, 5])
    pd.testing.assert_frame_equal(obs.obsdata, truth)


def test_plotobs_subset_time(sample_obs):
    """Test PlotObs subsetting for a particular time."""
    obs = PlotObs()
    obs.data = sample_obs
    obs.level = None
    obs.time = datetime(2020, 8, 6, 13)

    truth = pd.DataFrame([('2020-08-06 13:00', 'KDEN', 500, 7, 15)],
                         columns=['time', 'stid', 'pressure', 'temperature', 'dewpoint'])
    truth = truth.set_index(pd.to_datetime(truth['time']))
    pd.testing.assert_frame_equal(obs.obsdata, truth)


def test_plotobs_subset_time_window(sample_obs):
    """Test PlotObs subsetting for a particular time with a window."""
    # Test also using an existing index
    sample_obs['time'] = pd.to_datetime(sample_obs['time'])
    sample_obs.set_index('time')

    obs = PlotObs()
    obs.data = sample_obs
    obs.level = None
    obs.time = datetime(2020, 8, 5, 12)
    obs.time_window = timedelta(minutes=30)

    truth = pd.DataFrame([(datetime(2020, 8, 5, 12), 'KDEN', 500, 3, 11),
                          (datetime(2020, 8, 5, 12, 1), 'KOKC', 500, 4, 12)],
                         columns=['time', 'stid', 'pressure', 'temperature', 'dewpoint'])
    truth = truth.set_index('time')
    pd.testing.assert_frame_equal(obs.obsdata, truth)


def test_plotobs_subset_time_window_level(sample_obs):
    """Test PlotObs subsetting for a particular time with a window and a level."""
    # Test also using an existing index
    sample_obs['time'] = pd.to_datetime(sample_obs['time'])
    sample_obs.set_index('time')

    obs = PlotObs()
    obs.data = sample_obs
    obs.level = 1000 * units.hPa
    obs.time = datetime(2020, 8, 5, 12)
    obs.time_window = timedelta(minutes=30)

    truth = pd.DataFrame([(datetime(2020, 8, 5, 12), 'KDEN', 1000, 1, 9),
                          (datetime(2020, 8, 5, 12, 1), 'KOKC', 1000, 2, 10)],
                         columns=['time', 'stid', 'pressure', 'temperature', 'dewpoint'])
    truth = truth.set_index('time')
    pd.testing.assert_frame_equal(obs.obsdata, truth)


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.016)
def test_plotobs_units_with_formatter(ccrs, pandas_sfc):
    """Test using PlotObs with a field that both has units and a custom formatter."""
    # Catch warning from Pandas due to setting units
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        pandas_sfc.units = {'alti': 'inHg'}

    # Plot desired data
    obs = PlotObs()
    obs.data = pandas_sfc
    obs.time = datetime(1993, 3, 12, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['alti']
    obs.plot_units = ['hPa']
    obs.locations = ['NE']
    # Set a format for plotting MSLP
    obs.formats = [lambda v: format(v * 10, '.0f')[-3:]]
    obs.reduce_points = 0.75

    # Panel for plot with Map features
    panel = MapPanel()
    panel.layout = (1, 1, 1)
    panel.projection = 'lcc'
    panel.area = 'in'
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.panels = [panel]
    pc.size = (10, 10)

    pc.draw()
    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.081 if version_check('cartopy<0.23') else 0.025)
def test_declarative_sfc_obs(ccrs, pandas_sfc):
    """Test making a surface observation plot."""
    obs = PlotObs()
    obs.data = pandas_sfc
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
                               tolerance=0.075 if version_check('cartopy<0.23') else 0.)
def test_declarative_sfc_obs_args(ccrs, pandas_sfc):
    """Test making a surface observation plot with mpl arguments."""
    obs = PlotObs()
    obs.data = pandas_sfc
    obs.time = datetime(1993, 3, 12, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['tmpf']
    obs.colors = ['black']
    obs.mpl_args = {'fontsize': 12}

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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.016)
@needs_cartopy
def test_declarative_sfc_text(pandas_sfc):
    """Test making a surface observation plot with text."""
    obs = PlotObs()
    obs.data = pandas_sfc
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
    panel.layers = []
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (10, 10)
    pc.panels = [panel]

    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.081 if version_check('cartopy<0.23') else 0.)
def test_declarative_sfc_obs_changes(ccrs, pandas_sfc):
    """Test making a surface observation plot, changing the field."""
    obs = PlotObs()
    obs.data = pandas_sfc
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.171)
def test_declarative_colored_barbs(ccrs, pandas_sfc):
    """Test making a surface plot with a colored barb (gh-1274)."""
    obs = PlotObs()
    obs.data = pandas_sfc
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.314)
def test_declarative_sfc_obs_full(ccrs, pandas_sfc):
    """Test making a full surface observation plot."""
    obs = PlotObs()
    obs.data = pandas_sfc
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.522)
@needs_cartopy
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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.518)
@needs_cartopy
def test_declarative_upa_obs_convert_barb_units():
    """Test making a full upperair observation plot with barbs converting units."""
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
    panel.layers = []
    panel.plots = [obs]

    # Bringing it all together
    pc = PanelContainer()
    pc.size = (15, 10)
    pc.panels = [panel]

    pc.draw()

    obs.level = 300 * units.hPa

    return pc.figure


def test_attribute_error_time(ccrs, pandas_sfc):
    """Make sure we get a useful error when the time variable is not found."""
    pandas_sfc.rename(columns={'valid': 'vtime'}, inplace=True)

    obs = PlotObs()
    obs.data = pandas_sfc
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

    plt.close(pc.figure)


def test_attribute_error_station(ccrs, pandas_sfc):
    """Make sure we get a useful error when the station variable is not found."""
    pandas_sfc.rename(columns={'station': 'location'}, inplace=True)

    obs = PlotObs()
    obs.data = pandas_sfc
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

    plt.close(pc.figure)


@pytest.mark.mpl_image_compare(remove_text=True,
                               tolerance=0.082 if version_check('cartopy<0.23') else 0.)
def test_declarative_sfc_obs_change_units(ccrs):
    """Test making a surface observation plot."""
    data = parse_metar_file(get_test_data('metar_20190701_1200.txt', as_file_obj=False),
                            year=2019, month=7)

    obs = PlotObs()
    obs.data = data
    obs.time = datetime(2019, 7, 1, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['air_temperature']
    obs.colors = ['black']
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
                               tolerance=0.125 if version_check('cartopy<0.23') else 0.0)
def test_declarative_multiple_sfc_obs_change_units(ccrs):
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


@pytest.mark.mpl_image_compare(remove_text=False, tolerance=0.607)
@needs_cartopy
def test_declarative_title_fontsize():
    """Test adjusting the font size of a MapPanel's title text."""
    data = xr.open_dataset(get_test_data('NAM_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height_isobaric'
    contour.level = 300 * units.hPa
    contour.linewidth = 2
    contour.contours = list(range(0, 2000, 12))
    contour.scale = 1e-1

    panel = MapPanel()
    panel.area = (-124, -72, 20, 53)
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]
    panel.title = '300 mb Geopotential Height'
    panel.title_fontsize = 20

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=False,
                               tolerance=0.951 if version_check('cartopy<0.23') else 0.)
@needs_cartopy
def test_declarative_colorbar_fontsize():
    """Test adjusting the font size of a colorbar."""
    data = xr.open_dataset(get_test_data('GFS_test.nc', as_file_obj=False))

    cfill = FilledContourPlot()
    cfill.data = data
    cfill.field = 'Temperature_isobaric'
    cfill.level = 300 * units.hPa
    cfill.time = datetime(2010, 10, 26, 12)
    cfill.contours = list(range(210, 250, 2))
    cfill.colormap = 'BuPu'
    cfill.colorbar = 'horizontal'
    cfill.colorbar_fontsize = 'x-small'

    panel = MapPanel()
    panel.area = (-124, -72, 20, 53)
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [cfill]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.607)
@needs_cartopy
def test_declarative_station_plot_fontsize():
    """Test adjusting the font size for station plots in PlotObs."""
    data = parse_metar_file(get_test_data('metar_20190701_1200.txt',
                                          as_file_obj=False), year=2019, month=7)
    obs = PlotObs()
    obs.data = data
    obs.time = datetime(2019, 7, 1, 12)
    obs.time_window = timedelta(minutes=15)
    obs.level = None
    obs.fields = ['cloud_coverage', 'air_temperature', 'dew_point_temperature',
                  'air_pressure_at_sea_level', 'current_wx1_symbol']
    obs.plot_units = [None, 'degF', 'degF', None, None]
    obs.locations = ['C', 'NW', 'SW', 'NE', 'W']
    obs.formats = ['sky_cover', None, None, lambda v: format(v * 10, '.0f')[-3:],
                   'current_weather']
    obs.reduce_points = 3
    obs.vector_field = ['eastward_wind', 'northward_wind']
    obs.fontsize = 8

    panel = MapPanel()
    panel.area = 'centus'
    panel.projection = 'lcc'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [obs]

    pc = PanelContainer()
    pc.size = (8, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.607)
@needs_cartopy
def test_declarative_contour_label_fontsize():
    """Test adjusting the font size of contour labels."""
    data = xr.open_dataset(get_test_data('NAM_test.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Geopotential_height_isobaric'
    contour.level = 300 * units.hPa
    contour.linewidth = 2
    contour.contours = list(range(0, 2000, 12))
    contour.scale = 1e-1
    contour.clabels = True
    contour.label_fontsize = 'xx-large'

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


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.02)
@needs_cartopy
def test_declarative_raster():
    """Test making a raster plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    raster = RasterPlot()
    raster.data = data
    raster.colormap = 'viridis'
    raster.field = 'Temperature'
    raster.level = 700 * units.hPa

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline']
    panel.plots = [raster]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.02)
@needs_cartopy
def test_declarative_raster_options():
    """Test making a raster plot."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    raster = RasterPlot()
    raster.data = data
    raster.colormap = 'viridis'
    raster.field = 'Temperature'
    raster.level = 700 * units.hPa
    raster.mpl_args = {'alpha': 1, 'cmap': 'coolwarm'}

    panel = MapPanel()
    panel.area = 'us'
    panel.projection = 'lcc'
    panel.layers = ['coastline']
    panel.plots = [raster]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.607)
@needs_cartopy
def test_declarative_region_modifier_zoom_in():
    """Test that '+' suffix on area string properly decreases extent of map."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa

    panel = MapPanel()
    panel.area = 'sc++'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=0.377)
@needs_cartopy
def test_declarative_region_modifier_zoom_out():
    """Test that '-' suffix on area string properly expands extent of map."""
    data = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

    contour = ContourPlot()
    contour.data = data
    contour.field = 'Temperature'
    contour.level = 700 * units.hPa

    panel = MapPanel()
    panel.area = 'sc-'
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [contour]

    pc = PanelContainer()
    pc.size = (8.0, 8)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@needs_cartopy
def test_declarative_bad_area():
    """Test that a invalid string or tuple provided to the area trait raises an error."""
    panel = MapPanel()

    # Test for string that cannot be grouped into a region and a modifier by regex
    with pytest.raises(TraitError):
        panel.area = 'a$z+'

    # Test for string that is not in our list of areas
    with pytest.raises(TraitError):
        panel.area = 'PS'

    # Test for nonsense coordinates
    with pytest.raises(TraitError):
        panel.area = (136, -452, -65, -88)


def test_save():
    """Test that our saving function works."""
    pc = PanelContainer()
    fobj = BytesIO()
    pc.save(fobj, format='png')
    plt.close(pc.figure)

    fobj.seek(0)

    # Test that our file object had something written to it.
    assert fobj.read()


def test_show():
    """Test that show works properly."""
    pc = PanelContainer()
    with patch.object(plt, 'show', new_callable=PropertyMock) as show:
        pc.show()
        show.assert_called()

    plt.close(pc.figure)


@needs_cartopy
def test_panel():
    """Test the functionality of the panel property."""
    panel = MapPanel()

    pc = PanelContainer()
    pc.panels = [panel]

    assert pc.panel is panel

    pc.panel = panel
    assert pc.panel is panel


@needs_cartopy
def test_copy():
    """Test that the copy method works for all classes in `declarative.py`."""
    # Copies of plot objects
    objects = [ImagePlot(), ContourPlot(), FilledContourPlot(), RasterPlot(), BarbPlot(),
               PlotObs(), PlotGeometry()]

    for obj in objects:
        obj.time = datetime.now()
        copied_obj = obj.copy()
        assert obj is not copied_obj
        assert obj.time == copied_obj.time

    # Copies of MapPanel and PanelContainer
    obj = MapPanel()
    obj.title = 'Sample Text'
    copied_obj = obj.copy()
    assert obj is not copied_obj
    assert obj.title == copied_obj.title

    obj = PanelContainer()
    obj.size = (10, 10)
    copied_obj = obj.copy()
    assert obj is not copied_obj
    assert obj.size == copied_obj.size
    plt.close(obj.figure)

    # Copies of plots in MapPanels should not point to same location in memory
    obj = MapPanel()
    obj.plots = [PlotObs(), PlotGeometry(), BarbPlot(), FilledContourPlot(), ContourPlot(),
                 RasterPlot(), ImagePlot()]
    copied_obj = obj.copy()

    for i in range(len(obj.plots)):
        assert obj.plots[i] is not copied_obj.plots[i]


@pytest.mark.mpl_image_compare(remove_text=False, tolerance=0.607)
@needs_cartopy
def test_declarative_plot_geometry_polygons():
    """Test that `PlotGeometry` correctly plots MultiPolygon and Polygon objects."""
    from shapely.geometry import MultiPolygon, Polygon

    # MultiPolygons and Polygons to plot
    slgt_risk_polygon = MultiPolygon([Polygon(
        [(-87.43, 41.86), (-91.13, 41.39), (-95.24, 40.99), (-97.47, 40.4), (-98.39, 41.38),
         (-96.54, 42.44), (-94.02, 44.48), (-92.62, 45.48), (-89.49, 45.91), (-86.38, 44.92),
         (-86.26, 43.37), (-86.62, 42.45), (-87.43, 41.86), ]), Polygon(
        [(-74.02, 42.8), (-72.01, 43.08), (-71.42, 42.77), (-71.76, 42.29), (-72.73, 41.89),
         (-73.89, 41.93), (-74.4, 42.28), (-74.02, 42.8), ])])
    enh_risk_polygon = Polygon(
        [(-87.42, 43.67), (-88.44, 42.65), (-90.87, 41.92), (-94.63, 41.84), (-95.13, 42.22),
         (-95.23, 42.54), (-94.79, 43.3), (-92.81, 43.99), (-90.62, 44.55), (-88.51, 44.61),
         (-87.42, 43.67)])

    # Plot geometry, set colors and labels
    geo = PlotGeometry()
    geo.geometry = [slgt_risk_polygon, enh_risk_polygon]
    geo.stroke = ['#DDAA00', '#FF6600']
    geo.stroke_width = [1]
    geo.fill = None
    geo.labels = ['SLGT', 'ENH']
    geo.label_facecolor = ['#FFE066', '#FFA366']
    geo.label_edgecolor = ['#DDAA00', '#FF6600']
    geo.label_fontsize = 'large'

    # Place plot in a panel and container
    panel = MapPanel()
    panel.area = [-125, -70, 20, 55]
    panel.projection = 'lcc'
    panel.title = ' '
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.plots = [geo]

    pc = PanelContainer()
    pc.size = (12, 12)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=False, tolerance=2.985)
def test_declarative_plot_geometry_lines(ccrs):
    """Test that `PlotGeometry` correctly plots MultiLineString and LineString objects."""
    from shapely.geometry import LineString, MultiLineString

    # LineString and MultiLineString to plot
    irma_fcst = LineString(
        [(-52.3, 16.9), (-53.9, 16.7), (-56.2, 16.6), (-58.6, 17.0), (-61.2, 17.8),
         (-63.9, 18.7), (-66.8, 19.6), (-72.0, 21.0), (-76.5, 22.0)])
    irma_fcst_shadow = MultiLineString([LineString(
        [(-52.3, 17.15), (-53.9, 16.95), (-56.2, 16.85), (-58.6, 17.25), (-61.2, 18.05),
         (-63.9, 18.95), (-66.8, 19.85), (-72.0, 21.25), (-76.5, 22.25)]), LineString(
        [(-52.3, 16.65), (-53.9, 16.45), (-56.2, 16.35), (-58.6, 16.75), (-61.2, 17.55),
         (-63.9, 18.45), (-66.8, 19.35), (-72.0, 20.75), (-76.5, 21.75)])])

    # Plot geometry, set colors and labels
    geo = PlotGeometry()
    geo.geometry = [irma_fcst, irma_fcst_shadow]
    geo.fill = None
    geo.stroke = 'green'
    geo.labels = ['Irma', '+/- 0.25 deg latitude']
    geo.label_facecolor = None
    geo.mpl_args = {'linewidths': 1}

    # Place plot in a panel and container
    panel = MapPanel()
    panel.area = [-85, -45, 12, 25]
    panel.projection = ccrs.PlateCarree()
    panel.layers = ['coastline', 'borders', 'usstates']
    panel.title = 'Hurricane Irma Forecast'
    panel.plots = [geo]

    pc = PanelContainer()
    pc.size = (12, 12)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=False, tolerance=0.013)
def test_declarative_plot_geometry_fills(ccrs):
    """Test that `PlotGeometry` correctly plots MultiLineString and LineString objects."""
    from shapely.geometry import Polygon

    # MultiPolygons and Polygons to plot
    mdt_risk_polygon = Polygon(
        [(-87.07, 31.68), (-88.65, 30.96), (-91.83, 30.52), (-92.83, 30.82), (-93.18, 31.77),
         (-92.65, 33.31), (-92.64, 35.22), (-91.74, 36.16), (-88.84, 35.53), (-86.13, 34.5),
         (-85.45, 33.01), (-86, 32.05), (-87.07, 31.68)])
    high_risk_polygon = Polygon(
        [(-91.88, 32.99), (-89.82, 34.4), (-88.19, 34.21), (-87.72, 33.46), (-88.23, 32.48),
         (-90.53, 32), (-91.74, 32.07), (-91.88, 32.99)])

    # Plot geometry, set colors and labels
    geo = PlotGeometry()
    geo.geometry = [mdt_risk_polygon, high_risk_polygon]
    geo.fill = ['#E06666', '#EE99EE']
    geo.stroke = ['#E06666', '#EE99EE']
    geo.labels = None

    # Place plot in a panel and container
    panel = MapPanel()
    panel.area = [-120, -75, 25, 50]
    panel.projection = 'lcc'
    panel.title = ' '
    panel.plots = [geo]

    pc = PanelContainer()
    pc.size = (12, 12)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@pytest.mark.mpl_image_compare(remove_text=False, tolerance=1.900)
def test_declarative_plot_geometry_points(ccrs):
    """Test that `PlotGeometry` correctly plots Point and MultiPoint objects."""
    from shapely.geometry import MultiPoint, Point

    # Points and MultiPoints to plot
    irma_track = [Point(-74.7, 21.8), Point(-76.0, 22.0), Point(-77.2, 22.1)]
    irma_track_shadow = MultiPoint([
        Point(-64.7, 18.25), Point(-66.0, 18.85), Point(-67.7, 19.45), Point(-69.0, 19.85),
        Point(-70.4, 20.45), Point(-71.8, 20.85), Point(-73.2, 21.25), Point(-74.7, 21.55),
        Point(-76.0, 21.75), Point(-77.2, 21.85), Point(-78.3, 22.05), Point(-79.3, 22.45),
        Point(-80.2, 22.85), Point(-80.9, 23.15), Point(-81.3, 23.45), Point(-81.5, 24.25),
        Point(-81.7, 25.35), Point(-81.7, 26.55), Point(-82.2, 27.95), Point(-82.7, 29.35),
        Point(-83.5, 30.65), Point(-84.4, 31.65)])

    # Plot geometry, set colors and labels
    geo = PlotGeometry()
    geo.geometry = irma_track + [irma_track_shadow]
    geo.fill = 'blue'
    geo.stroke = None
    geo.marker = '^'
    geo.labels = ['Point', 'Point', 'Point', 'Irma Track']
    geo.label_edgecolor = None
    geo.label_facecolor = None

    # Place plot in a panel and container
    panel = MapPanel()
    panel.area = [-85, -65, 17, 30]
    panel.projection = ccrs.PlateCarree()
    panel.layers = ['states', 'coastline', 'borders']
    panel.plots = [geo]

    pc = PanelContainer()
    pc.size = (12, 12)
    pc.panels = [panel]
    pc.draw()

    return pc.figure


@needs_cartopy
def test_drop_traitlets_dir():
    """Test successful drop of inherited members from HasTraits and any '_' or '__' members."""
    for plot_obj in (
            BarbPlot,
            ContourPlot,
            FilledContourPlot,
            RasterPlot,
            ImagePlot,
            MapPanel,
            PanelContainer,
            PlotGeometry,
            PlotObs
    ):
        assert dir(plot_obj)[0].startswith('_')
        assert not dir(plot_obj())[0].startswith('_')
        assert 'cross_validation_lock' in dir(plot_obj)
        assert 'cross_validation_lock' not in dir(plot_obj())


@needs_cartopy
def test_attribute_error_suggest():
    """Test that a mistyped attribute name raises an exception with fix."""
    with pytest.raises(AttributeError) as excinfo:
        panel = MapPanel()
        panel.pots = []
    assert "Perhaps you meant 'plots'?" in str(excinfo.value)


@needs_cartopy
def test_attribute_error_no_suggest():
    """Test that a mistyped attribute name raises an exception w/o a fix."""
    with pytest.raises(AttributeError) as excinfo:
        panel = MapPanel()
        panel.galaxy = 'Andromeda'
    assert 'Perhaps you meant' not in str(excinfo.value)
