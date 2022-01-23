=============
Install Guide
=============

------------
Requirements
------------
In general, MetPy tries to support minor versions of dependencies released within the last two
years. For Python itself, that generally means supporting the last two minor releases; MetPy
currently supports Python >= 3.7.

.. literalinclude:: ../../setup.cfg
   :start-after: importlib_resources
   :end-at: xarray

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

------------------
Working With Conda
------------------

MetPy Monday videos `#1`_, `#2`_, and `#3`_ demonstrate how to install the conda package
manager and Python packages, and how to work with conda environments.

.. _#1: https://youtu.be/-fOfyHYpKck
.. _#2: https://youtu.be/G3AF-nhNyDk
.. _#3: https://youtu.be/15DNH25UCi0

--------
Examples
--------

The MetPy source comes with a set of example scripts in the ``examples``
directory. These are also available as notebooks in the :doc:`/examples/index`.
