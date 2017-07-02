==================
Installation Guide
==================

------------
Requirements
------------
MetPy supports Python 2.7 as well as Python >= 3.3. Python 3.4 is the recommended version.

MetPy requires the following packages:
  - NumPy >= 1.8.0
  - SciPy >= 0.13.3
  - Matplotlib >= 1.4.0
  - pint >= 0.6

Installation Instructions for NumPy and SciPy can be found at:
  http://www.scipy.org/scipylib/download.html

Installation Instructions for Matplotlib can be found at:
  http://matplotlib.org/downloads.html

Pint is a pure python package and can be installed via ``pip install pint``.

Python versions older than 3.4 require the enum34 package, which is a backport
of the enum standard library module. It can be installed via
``pip install enum34``.

PyProj is an optional dependency (if using the CDM interface to data files).
It can also be installed via ``pip install pyproj``, though it does require
the Proj.4 library and a compiled extension.

------------
Installation
------------

The easiest way to install MetPy is through ``pip``:

.. parsed-literal::
    pip install metpy

The source code can also be grabbed from `GitHub <http://github.com/MetPy/metpy>`_. From
the base of the source directory, run:

.. parsed-literal::
    python setup.py install

This will build and install MetPy into your current Python installation.

--------
Examples
--------

The MetPy source comes with a set of example IPython notebooks in the ``examples/notebooks`` directory.
These can also be converted to standalone scripts (provided IPython is installed) using:

.. parsed-literal::
    python setup.py examples

These examples are also seen within the documentation in the :ref:`examples-index`.
