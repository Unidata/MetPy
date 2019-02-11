# Copyright (c) 2008,2010,2015,2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Setup script for installing MetPy."""

from __future__ import print_function

from setuptools import find_packages, setup
import versioneer

ver = versioneer.get_version()

setup(
    name='MetPy',
    version=ver,
    description='Collection of tools for reading, visualizing and'
                'performing calculations with weather data.',
    long_description='The space MetPy aims for is GEMPAK '
                     '(and maybe NCL)-like functionality, in a way that '
                     'plugs easily into the existing scientific Python '
                     'ecosystem (numpy, scipy, matplotlib).',

    url='http://github.com/Unidata/MetPy',

    author='Ryan May, Patrick Marsh, Sean Arms, Eric Bruning',
    author_email='python-users@unidata.ucar.edu',
    maintainer='MetPy Developers',
    maintainer_email='python-users@unidata.ucar.edu',

    license='BSD',

    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Atmospheric Science',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'License :: OSI Approved :: BSD License'],
    keywords='meteorology weather',

    packages=find_packages(exclude=['doc', 'examples']),
    package_data={'metpy.plots': ['colortable_files/*.tbl', 'nexrad_tables/*.tbl',
                                  'fonts/*.ttf', '_static/metpy_75x75.png',
                                  '_static/metpy_150x150.png', '_static/unidata_75x75.png',
                                  '_static/unidata_150x150.png'],
                  'metpy': ['static-data-manifest.txt']},

    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*',
    install_requires=['matplotlib>=2.0.0', 'numpy>=1.12.0', 'scipy>=0.17.0',
                      'pint>=0.8', 'xarray>=0.10.7', 'enum34;python_version<"3.4"',
                      'contextlib2;python_version<"3.6"',
                      'pooch>=0.1', 'traitlets>=4.3.0'],
    extras_require={
        'cdm': ['pyproj>=1.9.4'],
        'dev': ['ipython[all]>=3.1'],
        'doc': ['sphinx>=1.4', 'sphinx-gallery', 'doc8', 'm2r',
                'netCDF4'],
        'examples': ['cartopy>=0.13.1', 'matplotlib>=2.2.0'],
        'test': ['pytest>=2.4', 'pytest-runner', 'pytest-mpl', 'pytest-flake8',
                 'cartopy>=0.16.0', 'flake8>3.2.0', 'flake8-builtins!=1.4.0',
                 'flake8-comprehensions', 'flake8-copyright',
                 'flake8-docstrings', 'flake8-import-order', 'flake8-mutable',
                 'flake8-pep3101', 'flake8-print', 'flake8-quotes',
                 'pep8-naming', 'netCDF4']
    },

    cmdclass=versioneer.get_cmdclass(),

    zip_safe=True,

    download_url='https://github.com/Unidata/MetPy/archive/v{}.tar.gz'.format(ver), )
