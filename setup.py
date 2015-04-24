from setuptools import setup, find_packages
import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'metpy/_version.py'
versioneer.versionfile_build = 'metpy/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = 'metpy-'


ver = versioneer.get_version()

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

    install_requires=['matplotlib>=1.4', 'numpy>=1.8', 'scipy>=0.14'],
    extras_require={
        'doc': ['sphinxcontrib-napoleon'],
        'test': ['nosetest']
    },

    cmdclass=versioneer.get_cmdclass(),

    download_url='https://github.com/metpy/MetPy/archive/v%s.tar.gz' % ver,)