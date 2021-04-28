=================
Developer's Guide
=================

.. toctree::
   :maxdepth: 3
   :hidden:

   CONTRIBUTING
   roadmap
   infrastructureguide

This discusses information relevant to developing MetPy.

--------
Versions
--------
MetPy follows `semantic versioning <https://semver.org>`_ in its version number. This means
that any MetPy ``1.x`` release will be backwards compatible with an earlier ``1.y`` release. By
"backward compatible", we mean that **correct** code that works on a ``1.y`` version will work
on a future ``1.x`` version. It's always possible for bug fixes to change behavior or make
incorrect code cease to work. Backwards-incompatible changes will only be allowed when changing
to version ``2.0``. Such changes will be proceeded by `MetpyDeprecationWarning` or
`FutureWarning` as appropriate. For a version ``1.x.y``, we change ``x`` when we
release new features, and ``y`` when we make a release with only bug fixes.

