====================
Infrastructure Guide
====================

This guide serves as an advanced version of the contributing documentation and contains the
information on how we manage MetPy and the infrastructure behind it.

----------
Versioning
----------

To manage identifying the version of the code, MetPy relies upon `setuptools_scm
<https://github.com/pypa/setuptools_scm>`_. ``setuptools_scm`` takes the current version of the
source from git tags and any additional commits. For development, the version will have a
string like ``0.10.0.post209+gff2e549f.d20190918``, which comes from ``git describe``. This
version means that the current code is 209 commits past the 0.10.0 tag, on git hash
``ff2e549f``, with local changes on top, last made on a date (indicated by ``d20190918``). For
a release, or non-git repo source dir, the version will just come from the most recent tag
(i.e. ``v0.10.0``). Our main versioning scheme matches ``vMajor.Minor.Bugfix``.

-------------
Documentation
-------------

MetPy's documentation is built using sphinx >= 2.1. API documentation is automatically
generated from docstrings, written using the
`NumPy docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
There are also example scripts in the ``examples`` directory, as well as our
:doc:`/userguide/index` tutorials in the ``tutorials``. Using the ``sphinx-gallery``
extension, these scripts are executed and turned into a gallery of thumbnails. The
extension also makes these scripts available as Jupyter notebooks.

The documentation is hosted on `GitHub Pages <https://unidata.github.io/MetPy>`_. The docs are
built automatically and uploaded upon pushes or merges to GitHub. Commits to ``master`` end up
in our development version docs, while commits to versioned branches will update the
docs for the corresponding version, which are located in the appropriately named subidrectory
on the ``gh-pages`` branch. We only maintain docs at the minor level, not the bugfix one.
The docs rely on the ``pydata-sphinx-theme`` package for styling the docs, which needs to be
installed for any local doc builds.

-----------
Other Tools
-----------

Continuous integration is performed by
`GitHub Actions <https://github.com/Unidata/MetPy/actions?query=workflow%3ACI>`_.
This integration runs the unit tests on Linux for all supported versions of Python, as well
as runs against the minimum package versions, using PyPI packages. This also runs against
a (non-exhaustive) matrix of python versions on macOS and Windows. In addition to these tests,
GitHub actions also builds the documentation and runs the examples across multiple platforms
and Python versions, as well as checks for any broken web links. ``flake8`` (along with a
variety of plugins found in ``ci/linting.txt``) is also run against the code to check
formatting using another job on GitHub Actions. As part of this linting job, the docs are also
checked using the ``doc8`` tool. Configurations for these are in a variety of files in
``.github/workflows``.

Test coverage is monitored by `codecov.io <https://codecov.io/github/Unidata/MetPy>`_.

The following services are used to track code quality:

* `Codacy <https://app.codacy.com/project/Unidata/MetPy/dashboard>`_
* `Code Climate <https://codeclimate.com/github/Unidata/MetPy>`_
* `Scrutinizer <https://scrutinizer-ci.com/g/Unidata/MetPy/?branch=master)>`_

---------
Releasing
---------

To create a new release, go to the GitHub page and make a new release. The tag should be a
sensible version number, like v1.0.0. Add a name (can just be the version) and add some release
notes on what the big changes are. It's also possible to use
`loghub <https://github.com/spyder-ide/loghub>`_ to get information on all the issues and PRs
that were closed for the relevant milestone.

~~~~
PyPI
~~~~

Once the new release is published on GitHub, this will create the tag, which will trigger
new builds GitHub actions (see ``release.yml``). This runs no tests, but assumes those were
all working before the release was officially tagged. This action takes care of building
PyPI packages (the wheel and source distribution) and uploads to PyPI.

To build and upload manually (if for some reason it is necessary):

1. Do a pull locally to grab the new tag. This will ensure that ``setuptools_scm`` will give
   you the proper version.
2. (optional) Perform a ``git clean -f -x -d`` from the root of the repository. This will
   **delete** everything not tracked by git, but will also ensure clean source distribution.
   ``MANIFEST.in`` is set to include/exclude mostly correctly, but could miss some things.
3. Run ``python setup.py sdist bdist_wheel`` (this requires that ``wheel`` is installed).
4. Upload using ``twine``: ``twine upload dist/*``, assuming the ``dist/`` directory contains
   only files for this release. This upload process will include any changes to the ``README``
   as well as any updated flags from ``setup.py``.

~~~~~
Conda
~~~~~

MetPy conda packages are automatically produced and uploaded to
`Anaconda.org <https://anaconda.org/conda-forge/MetPy>`_ thanks to conda-forge. Once the
release is built and uploaded to PyPI, conda-forge's automated bot infrastructure should
(within anywhere from 30 minutes to a couple hours) produce a Pull Request on the
`MetPy feedstock <https://github.com/conda-forge/metpy-feedstock>`_, which handles updating
the package. This repository contains the recipe for building MetPy's conda packages.

If for some reason the bots fail or are delayed, a PR for the version update can be done
manually. This should:

1. Update the version
2. Update the hash to match that of the new source distribution **uploaded to PyPI**
3. Reset the build number to 0 (if necessary)
4. Update the dependencies (and their versions) as necessary

The Pull Request will test building the packages on all the platforms. Once this succeeds,
the Pull Request can be merged, which will trigger the final build and upload of the
packages to anaconda.org.
