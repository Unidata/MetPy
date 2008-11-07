from distutils.core import setup

setup(name = 'MetPy',
      version = '0.1',
      packages = ['metpy', 'metpy.bl', 'metpy.bl.sim', 'metpy.bl.turb',
        'metpy.vis', 'metpy.getobs', 'metpy.mkobsnc'],
      platforms = ['Linux'],
      description = 'Collection of tools for reading, visualizing and'
        'performing calculations with weather data.',
      url = 'http://code.forwarn.org/metpy',
      )
