==========
User Guide
==========

.. toctree::
   :maxdepth: 3
   :hidden:

   installguide
   startingguide
   /tutorials/index
   upgradeguide
   gempak
   SUPPORT
   citing
   media

.. plot::
    :include-source: False
    :width: 300px
    :align: left

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import metpy.calc as mpcalc
    from metpy.cbook import get_test_data
    from metpy.plots import SkewT
    from metpy.units import units

    fig = plt.figure(figsize=(4, 4), dpi=150)
    skew = SkewT(fig, rotation=45)

    col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']

    df = pd.read_fwf(get_test_data('may4_sounding.txt', as_file_obj=False),
                     skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)

    # Drop any rows with all NaN values for T, Td, winds
    df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'), how='all'
                   ).reset_index(drop=True)

    p = df['pressure'].values * units.hPa
    t = df['temperature'].values * units.degC
    td = df['dewpoint'].values * units.degC
    wind_speed = df['speed'].values * units.knots
    wind_dir = df['direction'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)
    p_prof, t, td, prof = mpcalc.parcel_profile_with_lcl(p, t, td)

    skew.plot(p_prof, t, 'r')
    skew.plot(p_prof, td, 'g')
    skew.plot(p_prof, prof, 'k')  # Plot parcel profile
    skew.plot_barbs(p, u, v, y_clip_radius=0.11)

    skew.ax.set_title('Sounding for 00Z 4 May 1999')
    skew.ax.set_xlim(-10, 35)
    skew.ax.set_xlabel(f'Temperature ({t.units:~})')
    skew.ax.set_ylim(1000, 250)
    skew.ax.set_ylabel(f'Pressure ({p_prof.units:~})')

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.shade_cin(p_prof, t, prof)
    skew.shade_cape(p_prof, t, prof)

    plt.show()

.. plot::
    :include-source: False
    :width: 300px
    :align: right

    import matplotlib.pyplot as plt
    import numpy as np
    from metpy.cbook import get_test_data
    from metpy.io import Level3File
    from metpy.plots import colortables

    name = get_test_data(f'nids/KOUN_SDUS54_N0QTLX_201305202016', as_file_obj=False)
    f = Level3File(name)
    datadict = f.sym_block[0][0]
    data = f.map_data(datadict['data'])

    az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
    rng = np.linspace(0, f.max_range, data.shape[-1] + 1)
    xlocs = rng * np.sin(np.deg2rad(az[:, np.newaxis]))
    ylocs = rng * np.cos(np.deg2rad(az[:, np.newaxis]))

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(4, 4), dpi=150)
    norm, cmap = colortables.get_with_steps('NWSStormClearReflectivity', -20, 0.5)
    ax.pcolormesh(xlocs, ylocs, data, norm=norm, cmap=cmap)
    ax.set_aspect('equal', 'datalim')
    ax.set_xlim(-40, 20)
    ax.set_ylim(-30, 30)
    ax.set_title('KTLX Reflectivity at 2013/05/20 2016Z')

    plt.show()

.. raw:: html

    <div class="clearer"></div>

MetPy's User Guide can help show you how to use MetPy, walking through various parts of
MetPy and show you to other resources. See the :doc:`/api/index` for more information on
a particular function or feature in MetPy.

New to MetPy? Try the :doc:`Getting Started Guide<startingguide>`.

Need some more direct help? Check out our :doc:`support resources<SUPPORT>`.

Used MetPy in your research? We'd love for you to :doc:`cite us<citing>` in your publication.

.. plot::
    :include-source: False
    :width: 600px
    :align: center

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
