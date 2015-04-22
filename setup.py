from __future__ import print_function, division
import distutils.sysconfig
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.install_data import install_data
import os
import os.path
import sys

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
    import numpy as np
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

ext_modules = []
include_dirs = None
build_ext = None

setup(
    name             = 'MetPy',
    packages         = ['metpy', 'metpy.calc', 'metpy.io', 'metpy.plots'],
    version          = ver,
    ext_modules      = ext_modules,
    include_dirs     = include_dirs,
    maintainer       = 'Unidata',
    maintainer_email = 'support-python@unidata.ucar.edu',
    description      = 'Collection of tools for reading, visualizing and'
                       'performing calculations with weather data.',
    url              = 'http://github.com/MetPy/MetPy',
    download_url     = 'https://github.com/metpy/MetPy/archive/v%s.tar.gz' % ver,
    keywords         = ['meteorology', 'weather'],
    classifiers      = ['Development Status :: 3 - Alpha',
                        'Programming Language :: Python',
                        'Programming Language :: Python :: 2.7',
                        'Programming Language :: Python :: 3.3',
                        'Programming Language :: Python :: 3.4',
                        'Topic :: Scientific/Engineering',
                        'Intended Audience :: Science/Research',
                        'Operating System :: OS Independent',
                        'License :: OSI Approved :: BSD License'],)

if not version.release:
  # Remove __git_version__ so that if we run from local, an outdated version
  # won't be found
  if os.path.exists(version._git_file_path):
      os.remove(version._git_file_path)
  if os.path.exists(version._git_file_path + 'c'):
      os.remove(version._git_file_path + 'c') #The .pyc file

  cleanup('.', '.html')
  cleanup('.', '.pyc')
  if sys.argv[1] in ['install']:
      import shutil
      shutil.rmtree('build')
