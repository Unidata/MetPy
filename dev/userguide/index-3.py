from datetime import datetime, timedelta
import cartopy.crs as ccrs
import pandas as pd
from metpy.cbook import get_test_data
import metpy.plots as mpplots

data = pd.read_csv(get_test_data('SFC_obs.csv', as_file_obj=False),
                   infer_datetime_format=True, parse_dates=['valid'])

obs = mpplots.PlotObs()
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
obs.reduce_points = 1.5

panel = mpplots.MapPanel()
panel.layout = (1, 1, 1)
panel.area = 'east'
panel.projection = ccrs.PlateCarree()
panel.layers = ['coastline', 'borders', 'states']
panel.plots = [obs]
panel.title = 'Surface Data Analysis'

pc = mpplots.PanelContainer()
pc.size = (12, 9)
pc.panels = [panel]

pc.show()