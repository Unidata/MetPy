from __future__ import print_function
from setuptools import setup, find_packages, Command
import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'metpy/_version.py'
versioneer.versionfile_build = 'metpy/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = 'metpy-'


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
        from IPython.nbconvert.exporters import python
        from IPython.config import Config
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


setup(
    name='MetPy',
    version=ver,
    description='Collection of tools for reading, visualizing and'
                'performing calculations with weather data.',
    url='http://github.com/MetPy/MetPy',

    maintainer='Unidata',
    maintainer_email='support-python@unidata.ucar.edu',

    license='BSD',

    classifiers=['Development Status :: 3 - Alpha',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4',
                 'Topic :: Scientific/Engineering',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'License :: OSI Approved :: BSD License'],
    keywords='meteorology weather',

    packages=find_packages(exclude=['doc', 'examples']),
    test_suite="nose.collector",

    install_requires=['matplotlib>=1.4', 'numpy>=1.8', 'scipy>=0.13.3',
                      'pint>=0.6'],
    extras_require={
        'dev': ['ipython[all]>=3.0'],
        'doc': ['sphinxcontrib-napoleon'],
        'test': ['nosetest']
    },

    cmdclass=commands,

    download_url='https://github.com/metpy/MetPy/archive/v%s.tar.gz' % ver,)
