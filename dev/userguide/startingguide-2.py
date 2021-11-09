import xarray as xr
from metpy.cbook import get_test_data
from metpy.plots import ImagePlot, MapPanel, PanelContainer
from metpy.units import units

# Use sample NARR data for plotting
narr = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj=False))

img = ImagePlot()
img.data = narr
img.field = 'Geopotential_height'
img.level = 850 * units.hPa

panel = MapPanel()
panel.area = 'us'
panel.layers = ['coastline', 'borders', 'states', 'rivers', 'ocean', 'land']
panel.title = 'NARR Example'
panel.plots = [img]

pc = PanelContainer()
pc.size = (10, 8)
pc.panels = [panel]
pc.show()