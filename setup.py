# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function
import sys
from setuptools import setup, find_packages, Command
import versioneer


class MakeExamples(Command):
    description = 'Create example scripts from IPython notebooks'
    user_options=[]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import os
        import os.path
        from nbconvert.exporters import python
        from traitlets.config import Config
        examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
        script_dir = os.path.join(examples_dir, 'scripts')
        if not os.path.exists(script_dir):
            os.makedirs(script_dir)
        c = Config({'Exporter': {'template_file': 'examples/python-scripts.tpl'}})
        exporter = python.PythonExporter(config=c)
        for fname in glob.glob(os.path.join(examples_dir, 'notebooks', '*.ipynb')):
            output, _ = exporter.from_filename(fname)
            out_fname = os.path.splitext(os.path.basename(fname))[0]
            out_name = os.path.join(script_dir, out_fname + '.py')
            print(fname, '->', out_name)
            with open(out_name, 'w') as outf:
                outf.write(output)


ver = versioneer.get_version()
commands = versioneer.get_cmdclass()
commands.update(examples=MakeExamples)

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

    url='http://github.com/MetPy/MetPy',

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
        'doc': ['sphinx>=1.3', 'ipython[all]>=3.1'],
        'examples': ['cartopy>=0.13.1'],
        'test': ['pytest', 'pytest-runner']
    },

    cmdclass=commands,

    zip_safe=True,

    download_url='https://github.com/metpy/MetPy/archive/v%s.tar.gz' % ver,)
