==================
Installation Guide
==================

------------
Requirements
------------
In general, MetPy tries to support minor versions of dependencies released within the last two
years. For Python itself, that means supporting the last two minor releases.

* matplotlib >= 2.1.0
* numpy >= 1.16.0
* scipy >= 1.0.0
* pint >= 0.10.1
* pandas >= 0.22.0
* xarray >= 0.13.0
* traitlets >= 4.3.0
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
the :doc:`examples/index`. Further examples of MetPy usage are available
in the `Unidata Python Gallery <https://unidata.github.io/python-gallery/>`_.
