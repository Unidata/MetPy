from distutils.core import setup

setup(name = 'MetPy',
      version = '0.1',
      packages = ['metpy', 'metpy.bl', 'metpy.readers', 'metpy.tools',
        'metpy.vis'],
      platforms = ['Linux'],
      description = 'Collection of tools for reading, visualizing and'
        'performing calculations with weather data.',
      url = 'http://code.forwarn.org/metpy',
      )
