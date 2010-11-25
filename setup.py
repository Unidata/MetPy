from __future__ import print_function, division
import distutils.sysconfig
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.install_data import install_data
import numpy as np
import os, sys

build_gauss = True

# Scan directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

# Scan directory for extension files, removing compiled
# or html files
def cleanup(dir, fext):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(fext):
            os.remove(path)
        elif os.path.isdir(path):
            cleanup(path, fext)


# Generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [np.get_include(), "."],   # the '.' is CRUCIAL!!
        extra_compile_args = ["-O3", "-Wall"],
        extra_link_args = [],
        libraries = [],
        )


setup_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(setup_path, 'metpy'))
import version
version.write_git_version()
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

setup(
    name            = 'MetPy',
    version         = ver,
    packages        = ['metpy', 'metpy.bl', 'metpy.readers', 'metpy.tools',
                       'metpy.vis'],
    ext_modules     = ext_modules,
    include_dirs    = include_dirs,
    cmdclass        = {'build_ext':build_ext},
    platforms       = ['Linux'],
    description     = 'Collection of tools for reading, visualizing and'
                      'performing calculations with weather data.',
    url             = 'http://code.forwarn.org/metpy',)

if not version.release:
  # Remove __git_version__ so that if we run from local, an outdated version
  # won't be found
  os.remove(version._git_file_path)
  os.remove(version._git_file_path + 'c') #The .pyc file

  cleanup('.', '.html')
  cleanup('.', '.pyc')
  if sys.argv[1] in ['install']:
      import shutil
      shutil.rmtree('build')
