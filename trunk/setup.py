from distutils.core import setup
import os.path, sys

setup_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(setup_path, 'metpy'))
import version
version.write_svn_version()
ver = version.get_version()
sys.path.pop()

setup(name = 'MetPy',
      version = ver,
      packages = ['metpy', 'metpy.bl', 'metpy.readers', 'metpy.tools',
        'metpy.vis'],
      platforms = ['Linux'],
      description = 'Collection of tools for reading, visualizing and'
        'performing calculations with weather data.',
      url = 'http://code.forwarn.org/metpy',
      )

if not version.release:
  # Remove __svn_version__ so that if we run from local, an outdated version
  # won't be found
  os.remove(version._svn_file_path)
  os.remove(version._svn_file_path + 'c') #The .pyc file
