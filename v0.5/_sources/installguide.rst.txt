==================
Installation Guide
==================

------------
Requirements
------------
MetPy supports Python 2.7 as well as Python >= 3.4. Python 3.5 is the recommended version.

MetPy requires the following packages:
  - NumPy >= 1.9.1
  - SciPy >= 0.14.0
  - Matplotlib >= 1.4.0
  - pint >= 0.7

Installation Instructions for NumPy and SciPy can be found at:
  https://www.scipy.org/scipylib/download.html

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

If you are a user of the `Conda <https://conda.io/docs/>`_ package manager, there are also
up-to-date packages for MetPy (as well as its dependencies) available from the
`conda-forge <https://conda-forge.github.io>`_ channel:

.. parsed-literal::
    conda install -c conda-forge metpy

The source code can also be grabbed from `GitHub <https://github.com/Unidata/MetPy>`_. From
the base of the source directory, run:

.. parsed-literal::
    pip install .

This will build and install MetPy into your current Python installation.

--------
Examples
--------

The MetPy source comes with a set of example scripts in the ``examples``
directory. These are also available as notebooks in the gallery in
the :doc:`examples/index`.
