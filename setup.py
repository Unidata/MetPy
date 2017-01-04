# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Setup script for installing MetPy."""

from __future__ import print_function

import sys

from setuptools import find_packages, setup
import versioneer

ver = versioneer.get_version()

# Need to conditionally add enum support for older Python
dependencies = ['matplotlib>=1.4', 'numpy>=1.9.1', 'scipy>=0.14', 'pint>=0.6']
if sys.version_info < (3, 4):
    dependencies.append('enum34')

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

    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'License :: OSI Approved :: BSD License'],
    keywords='meteorology weather',

    packages=find_packages(exclude=['doc', 'examples']),
    package_data={'metpy.plots': ['colortables/*.tbl', 'nexrad_tables/*.tbl',
                                  'fonts/*.ttf']},

    install_requires=dependencies,
    extras_require={
        'cdm': ['pyproj>=1.9.4'],
        'dev': ['ipython[all]>=3.1'],
        'doc': ['sphinx>=1.4', 'sphinx-gallery', 'doc8'],
        'examples': ['cartopy>=0.13.1'],
        'test': ['pytest>=2.4', 'pytest-runner', 'pytest-mpl', 'pytest-flake8',
                 'flake8>3.2.0', 'flake8-builtins',
                 'flake8-comprehensions', 'flake8-copyright',
                 'flake8-docstrings', 'flake8-import-order', 'flake8-mutable',
                 'flake8-pep3101', 'flake8-print', 'flake8-quotes',
                 'pep8-naming']
    },

    cmdclass=versioneer.get_cmdclass(),

    zip_safe=True,

    download_url='https://github.com/Unidata/MetPy/archive/v{}.tar.gz'.format(ver),)
