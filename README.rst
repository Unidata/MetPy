MetPy
=====

.. image:: https://img.shields.io/pypi/l/metpy.svg
    :target: https://pypi.python.org/pypi/MetPy/
    :alt: License

.. image:: https://img.shields.io/github/issues/metpy/metpy.svg
    :target: http://www.github.com/metpy/MetPy/issues
    :alt: GitHub Issues

.. image:: https://img.shields.io/github/tag/metpy/metpy.svg
    :target: https://github.com/metpy/MetPy/tags
    :alt: GitHub Tags

.. image:: https://img.shields.io/pypi/v/metpy.svg
    :target: https://pypi.python.org/pypi/MetPy/
    :alt: PyPI Package

.. image:: https://img.shields.io/pypi/dm/metpy.svg
    :target: https://pypi.python.org/pypi/MetPy/
    :alt: PyPI Downloads

.. image:: https://binstar.org/unidata/metpy/badges/version.svg
    :target: https://binstar.org/unidata/metpy
    :alt: Binstar Package

.. image:: https://binstar.org/unidata/metpy/badges/downloads.svg
    :target: https://binstar.org/unidata/metpy
    :alt: Binstar Downloads

.. image:: https://travis-ci.org/metpy/MetPy.svg?branch=master
    :target: https://travis-ci.org/metpy/MetPy
    :alt: Travis Build Status

.. image:: https://codecov.io/github/metpy/MetPy/coverage.svg?branch=master
    :target: https://codecov.io/github/metpy/MetPy?branch=master
    :alt: Code Coverage Status

.. image:: https://www.quantifiedcode.com/api/v1/project/1153e58350aa41e6a7970a134febeb2d/badge.svg
    :target: https://www.quantifiedcode.com/app/project/1153e58350aa41e6a7970a134febeb2d
    :alt: Code issues

.. image:: https://api.codacy.com/project/badge/grade/e1ea0937eb4942e79a44bc9bb2de616d
    :target: https://www.codacy.com/app/dopplershift/MetPy
    :alt: Codacy code issues

.. image:: https://readthedocs.org/projects/pip/badge/?version=latest
    :target: http://metpy.readthedocs.org/en/latest/
    :alt: Latest Doc Build Status

.. image:: https://readthedocs.org/projects/pip/badge/?version=stable
    :target: http://metpy.readthedocs.org/en/stable/
    :alt: Stable Doc Build Status

MetPy is a collection of tools in Python for reading, visualizing and
performing calculations with weather data.

MetPy is still in an early stage of development, and as such
**no APIs are considered stable.** While we won't break things
just for fun, many things may still change as we work through
design issues.

We support Python 2.7 as well as Python >= 3.2.

Important Links
---------------

- Source code repository: https://github.com/MetPy/MetPy
- HTML Documentation (stable release): http://metpy.readthedocs.org/en/stable/
- HTML Documentation (development): http://metpy.readthedocs.org/en/latest/
- Issue tracker: http://github.com/Metpy/MetPy/issues

Dependencies
------------
Other required packages:

- Numpy
- Scipy
- Matplotlib
- Pint

Python versions older than 3.4 require the enum34 package, which is a backport
of the standard library enum module.

There is also an optional dependency on the pyproj library for geographic
projections (used with CDM interface).

Philosophy
----------
The space MetPy aims for is GEMPAK (and maybe NCL)-like functionality, in a way that plugs easily
into the existing scientific Python ecosystem (numpy, scipy, matplotlib). So, if you take the average GEMPAK script
for a weather map, you need to:

- read data
- calculate a derived field
- show on a map/skew-T

One of the benefits hoped to achieve over GEMPAK is to make it easier to use these routines for any
meteorological Python application; this means making it easy to pull out the LCL calculation and just use that,
or re-use the Skew-T with your own data code. MetPy also prides itself on being well-documented and well-tested,
so that on-going maintenance is easily manageable.

The intended audience is that of GEMPAK: researchers, educators, and any one wanting to script up weather analysis.
It doesn't even have to be scripting; all python meteorology tools are hoped to be able to benefit from MetPy.
Conversely, it's hoped to be the meteorological equivalent of the audience of scipy/scikit-learn/skimage.
