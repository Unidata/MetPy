=================
Developer's Guide
=================

------------
Requirements
------------

- pytest
- flake8
- sphinx >= 1.3
- sphinx-rtd-theme >= 0.1.7
- IPython >= 3.1
- pandoc (not a python package)

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

The changes to the MetPy source (and documentation) should be made via GitHub pull
requestsagainst ``master``, even for those with administration rights. While it's tempting to
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
<https://www.python.org/dev/peps/pep-0008/>`_. For better or worse, this is what the majority
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

MetPy's documentation is built using sphinx >= 1.3. API documentation is automatically
generated from docstrings, written using the
`NumPy docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
There are also example IPython notebooks in the ``examples/notebooks`` directory. Using
IPython's API, these are automatically converted to restructured text for inclusion in the
documentation. The examples can also be converted to standalone scripts using:

.. parsed-literal::
    python setup.py examples

The documentation is hosted by `Read the Docs <http://metpy.readthedocs.org>`_. The docs are
built automatically from ``master`` as well as for the tagged versions on github. ``master`` is
used for the ``latest`` documentation, and the latest tagged version is used for the ``stable``
documentation. To see what the docs will look like on RTD, you also need to install the
``sphinx-rtd-theme`` package.

-----------
Other Tools
-----------

Continuous integration is performed by `Travis CI <http://www.travis-ci.org/metpy/MetPy>`_.
This service runs the unit tests on all support versions, as well as runs against the minimum
package versions. ``flake8`` is also run against the code to check formatting. Travis is also
used to build the documentation and to run the examples to ensure they stay working.

Test coverage is monitored by `codecov.io <https://codecov.io/github/metpy/MetPy>`_.

The following services are used to track code quality:
* `QuantifiedCode <https://www.quantifiedcode.com/app/project/gh:metpy:MetPy>`_
* `Codacy <https://www.codacy.com/app/dopplershift/MetPy/dashboard>`_
* `Landscape.io <https://landscape.io/github/metpy/MetPy>`_

---------
Releasing
---------

To create a new release:

1. Go to the GitHub page and make a new release. The tag should be a sensible version number,
   like v1.0.0. Add a name (can just be the version) and add some notes on what the big
   changes are.
2. Do a pull locally to grab the new tag. This will ensure that ``versioneer`` will give you
   the proper version.
3. (optional) Perform a ``git clean -f -x -d`` from the root of the repository. This will
   **delete** everything not tracked by git, but will also ensure clean source distribution.
   ``MANIFEST.in`` is set to include/exclude mostly correctly, but could miss some things.
4. Run ``python setup.py sdist bdist_wheel`` (this requires ``wheel`` is installed).
5. Upload using ``twine``: ``twine upload dist/*``, assuming the ``dist/`` directory contains
   only files for this release. This upload process will include any changes to the ``README``
   as well as any updated flags from ``setup.py``.
6. Tagging a new version on GitHub should also update the
   `stable <http://metpy.readthedocs.org/en/stable>`_  docs on Read the Docs.
