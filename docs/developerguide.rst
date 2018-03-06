=================
Developer's Guide
=================

------------
Requirements
------------

- pytest >= 2.4
- flake8
- sphinx >= 1.3
- sphinx-rtd-theme >= 0.1.7
- nbconvert>=4.1

~~~~~
Conda
~~~~~

Settings up a development environment in MetPy is as easy as (from the
base of the repository):

.. parsed-literal::
    conda env create
    conda develop -n devel .

The ``environment.yml`` contains all of the configuration needed to easily
set up the environment, called ``devel``. The second line sets up conda to
run directly out of the git repository.

--------------
Making Changes
--------------

The changes to the MetPy source (and documentation) should be made via GitHub pull requests
against ``master``, even for those with administration rights. While it's tempting to
make changes directly to ``master`` and push them up, it is better to make a pull request so
that others can give feedback. If nothing else, this gives a chance for the automated tests to
run on the PR. This can eliminate "brown paper bag" moments with buggy commits on the master
branch.

During the Pull Request process, before the final merge, it's a good idea to rebase the branch
and squash together smaller commits. It's not necessary to flatten the entire branch, but it
can be nice to eliminate small fixes and get the merge down to logically arranged commit. This
can also be used to hide sins from history--this is the only chance, since once it hits
``master``, it's there forever!

----------
Versioning
----------

To manage identifying the version of the code, MetPy relies upon `versioneer
<https://github.com/warner/python-versioneer>`_. ``versioneer`` takes the current version of
the source from git tags and any additional commits. For development, the version will have a
string like ``0.1.1+76.g136e37b.dirty``, which comes from ``git describe``. This version means
that the current code is 76 commits past the 0.1.1 tag, on git hash ``136e37b``, with local
changes on top (indicated by ``dirty``). For a release, or non-git repo source dir, the version
will just come from the most recent tag (i.e. ``v0.1.1``).

To make a new version, simply add a new tag with a name like ``vMajor.Minor.Bugfix`` and push
to GitHub. Github will add a new release with a source archive.zip file. Running

.. parsed-literal::
    python setup.py sdist

will build a new source distribution with the appropriately generated version file as well.
This will also create a new stable set of documentation.

``versioneer`` is installed in the base of the repository. To update, install the latest copy
using ``pip install versioneer``. Then recreate the ``_version.py`` file using:

.. parsed-literal::
    python setup.py versioneer

-------
Testing
-------

Unit tests are the lifeblood of the project, as it ensures that we can continue to add and
change the code and stay confident that things have not broken. Running the tests requires
``pytest``, which is easily available through ``conda`` or ``pip``. Running the tests can be
done via either:

.. parsed-literal::
    python setup.py test

or

.. parsed-literal::
    py.test

Using ``py.test`` also gives you the option of passing a path to the directory with tests to
run, which can speed running only the tests of interest when doing development. For instance,
to only run the tests in the ``metpy/calc`` directory, use:

.. parsed-literal::
    py.test metpy/calc

Some tests (for matplotlib plotting code) are done through an image comparison, using the
pytest-mpl plugin. To run these tests, use:

.. parsed-literal::
    py.test --mpl

When adding new image comparison tests, start by creating the baseline images for the tests:

.. parsed-literal::
    py.test --mpl-generate-path=baseline

That command runs the tests and saves the images in the ``baseline`` directory. Once the images
are reviewed and determined to be correct, they should be moved to a ``baseline`` directory in
the same directory as the test script (e.g. ``metpy/plots/tests``) For more information, see
the `docs for mpl-test <https://github.com/astrofrog/pytest-mpl>`_.

----------
Code Style
----------

MetPy uses the Python code style outlined in `PEP8
<http://pep8.org>`_. For better or worse, this is what the majority
of the Python world uses. The one deviation is that line length limit is 95 characters. 80 is a
good target, but some times longer lines are needed.

While the authors are no fans of blind adherence to style and so-called project "clean-ups"
that go through and correct code style, MetPy has adopted this style from the outset.
Therefore, it makes sense to enforce this style as code is added to keep everything clean and
uniform. To this end, part of the automated testing for MetPy checks style. To check style
locally within the source directory you can use the ``flake8`` tool. Running it
from the root of the source directory is as easy as:

.. parsed-literal::
    flake8 metpy

-------------
Documentation
-------------

MetPy's documentation is built using sphinx >= 1.4. API documentation is automatically
generated from docstrings, written using the
`NumPy docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
There are also example scripts in the ``examples`` directory. Using the ``sphinx-gallery``
extension, these examples are executed and turned into a gallery of thumbnails. The
extension also makes these scripts available as Jupyter notebooks.

The documentation is hosted on `GitHub Pages <https://unidata.github.io/MetPy>`_. The docs are
built automatically from ``master`` with every build on Travis-CI; every merged PR will
have the built docs upload to GitHub Pages. As part of the build, the documentation is also
checked with ``doc8``. To see what the docs will look like, you also need to install the
``sphinx-rtd-theme`` package.

-----------
Other Tools
-----------

Continuous integration is performed by `Travis CI <http://www.travis-ci.org/Unidata/MetPy>`_
and `AppVeyor <https://ci.appveyor.com/project/Unidata/metpy/branch/master>`_.
Travis runs the unit tests on Linux for all supported versions of Python, as well as runs
against the minimum package versions. ``flake8`` (with the ``pep8-naming`` and
``flake8-quotes`` plugins) is also run against the code to check formatting. Travis is also
used to build the documentation and to run the examples to ensure they stay working. AppVeyor
is a similar service; here the tests and examples are run against Python 2 and 3 for both
32- and 64-bit versions of Windows.

Test coverage is monitored by `codecov.io <https://codecov.io/github/Unidata/MetPy>`_.

The following services are used to track code quality:

* `Codacy <https://www.codacy.com/app/Unidata/MetPy/dashboard>`_
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
new builds on Travis (and AppVeyor, but that's not relevant). When the main test build on
Travis (currently Python 3 tests) succeeds, Travis will handle building the source
distribution and wheels, and upload them to PyPI.

To build and upload manually (if for some reason it is necessary):

1. Do a pull locally to grab the new tag. This will ensure that ``versioneer`` will give you
   the proper version.
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
release is built and uploaded to PyPI, then a Pull Request should be made against the
`MetPy feedstock <https://github.com/conda-forge/metpy-feedstock>`_, which contains the
recipe for building MetPy's conda packages. The Pull Request should:

1. Update the version
2. Update the hash to match that of the new source distribution **uploaded to PyPI**
3. Reset the build number to 0 (if necessary)
4. Update the dependencies (and their versions) as necessary

The Pull Request will test building the packages on all the platforms. Once this succeeds,
the Pull Request can be merged, which will trigger the final build and upload of the
packages to anaconda.org.
