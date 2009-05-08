from distutils.core import setup
import os.path, sys

build_gauss = False

setup_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(setup_path, 'metpy'))
import version
version.write_svn_version()
ver = version.get_version()
sys.path.pop()

# Optionally build the gaussian filter code
if build_gauss:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    include_dirs = get_numpy_include_dirs()
    from distutils.extension import Extension

    # If we find Cython, build from the pyx file, otherwise just build the
    # C file
    try:
        from Cython.Distutils import build_ext
        files = ["src/gauss.pyx"]
    except ImportError:
        from distutils.command.build_ext import build_ext
        files = ["src/gauss.c"]

    ext_modules = [Extension("metpy.tools._gauss_filt", files,
        extra_compile_args=['-O2 -fomit-frame-pointer'])]
else:
    ext_modules = []
    include_dirs = None
    build_ext = None

setup(name = 'MetPy',
      version = ver,
      packages = ['metpy', 'metpy.bl', 'metpy.readers', 'metpy.tools',
        'metpy.vis'],
      ext_modules = ext_modules,
      include_dirs = include_dirs,
      cmdclass = {'build_ext':build_ext},
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
