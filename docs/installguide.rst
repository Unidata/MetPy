==================
Installation Guide
==================

.. _python27:

------------------
Python 2.7 Support
------------------
In the Fall 2019, we will be dropping support for Python 2.7. This follows movement from
other packages within the `scientific Python ecosystem <http://python3statement.org/>`_.
This includes:

* Core Python developers will
  `stop support for Python 2.7 January 1, 2020 <https://pythonclock.org/>`_
* NumPy feature releases will be
  `Python 3 only starting January 1, 2019 <https://www.numpy.org/neps/nep-0014-dropping-python2.7-proposal.html>`_,
  and support for the last release supporting Python 2 will end January 1, 2020.
* XArray will drop
  `2.7 January 1, 2019 as well <https://github.com/pydata/xarray/issues/1830>`_
* Matplotlib's 3.0 release, tentatively Summer 2018,
  `will be Python 3 only <https://mail.python.org/pipermail/matplotlib-devel/2017-October/000892.html>`_;
  the current 2.2 release will be the last long term release that supports 2.7, and its support
  will cease January 1, 2020.

The last release of MetPy before this time (Spring or Summer 2019) will be the last that
support Python 2.7. This version of MetPy will **not** receive any long term support or
additional bug fix releases after the next minor release. The packages for this version *will*
remain available on Conda or PyPI.

------------
Requirements
------------
In general, MetPy tries to support minor versions of dependencies released within the last two
years. For Python itself, that means supporting the last two minor releases, as well as
currently supporting Python 2.7.

* matplotlib >= 2.0.0
* numpy >= 1.12.0
* scipy >= 0.17.0
* pint >= 0.8
* xarray >= 0.10.7
* traitlets >= 4.3.0
* enum34 (for python < 3.4)
* pooch >= 0.1

------------
Installation
------------

The easiest way to install MetPy is through ``pip``:

.. parsed-literal::
    pip install metpy

If you are a user of the `Conda <https://conda.io/docs/>`_ package manager, there are also
up-to-date packages for MetPy (as well as its dependencies) available from the
`conda-forge <https://conda-forge.org>`_ channel:

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
