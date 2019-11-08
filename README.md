MetPy
=====

[![License](https://img.shields.io/pypi/l/metpy.svg)](https://pypi.python.org/pypi/MetPy/)
[![Gitter](https://badges.gitter.im/Unidata/MetPy.svg)](https://gitter.im/Unidata/MetPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=round-square)](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github)

[![Latest Docs](https://img.shields.io/badge/docs-stable-brightgreen.svg)](http://unidata.github.io/MetPy)
[![PyPI Package](https://img.shields.io/pypi/v/metpy.svg)](https://pypi.python.org/pypi/MetPy/)
[![Conda Package](https://anaconda.org/conda-forge/metpy/badges/version.svg)](https://anaconda.org/conda-forge/metpy)

[![PyPI Downloads](https://img.shields.io/pypi/dm/metpy.svg)](https://pypi.python.org/pypi/MetPy/)
[![Conda Downloads](https://anaconda.org/conda-forge/metpy/badges/downloads.svg)](https://anaconda.org/conda-forge/metpy)

[![Travis Build Status](https://travis-ci.com/Unidata/MetPy.svg?branch=master)](https://travis-ci.com/Unidata/MetPy)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/dwaletlb23v2ae4e/branch/master?svg=true)](https://ci.appveyor.com/project/Unidata/metpy/branch/master)
[![Code Coverage Status](https://codecov.io/github/Unidata/MetPy/coverage.svg?branch=master)](https://codecov.io/github/Unidata/MetPy?branch=master)

[![Codacy issues](https://api.codacy.com/project/badge/Grade/e1ea0937eb4942e79a44bc9bb2de616d)](https://www.codacy.com/app/dopplershift/MetPy)
[![Code Climate](https://codeclimate.com/github/Unidata/MetPy/badges/gpa.svg)](https://codeclimate.com/github/Unidata/MetPy)

MetPy is a collection of tools in Python for reading, visualizing and
performing calculations with weather data.

MetPy follows [semantic versioning](https://semver.org) in its version number. With our
current 0.x version, that implies that MetPy's APIs (application programming interfaces) are
still evolving (we won't break things just for fun, but many things are still changing as we
work through design issues). Also, for a version `0.x.y`, we change `x` when we
release new features, and `y` when we make a release with only bug fixes.

For additional MetPy examples not included in this repository, please see the [Unidata Python
Gallery](https://unidata.github.io/python-gallery/).

We support Python >= 3.6.

0.12 is the first version to drop support for Python 2.7 in Fall 2019. See
`here <https://github.com/Unidata/MetPy/blob/master/docs/installguide.rst>`_ for more
information.

Need Help?
----------
Need help using MetPy? Found an issue? Have a feature request? Checkout our
[support page](https://github.com/Unidata/MetPy/blob/master/SUPPORT.md).

Important Links
---------------

- [HTML Documentation](http://unidata.github.io/MetPy)
- [Unidata Python Gallery](https://unidata.github.io/python-gallery/)
- "metpy" tagged questions on [Stack Overflow](https://stackoverflow.com/questions/tagged/metpy)
- [Gitter chat room](https://gitter.im/Unidata/MetPy)
- [Say Thanks!](https://saythanks.io/to/unidata)

Dependencies
------------
Other required packages:

- Numpy
- Scipy
- Matplotlib
- Pandas
- Pint
- Xarray

There is also an optional dependency on the pyproj library for geographic
projections (used with cross sections, grid spacing calculation, and the GiniFile interface).

See the [installation guide](https://unidata.github.io/MetPy/dev/installguide.html)
for more information.

Code of Conduct
---------------
We want everyone to feel welcome to contribute to MetPy and participate in discussions. In that
spirit please have a look at our [Code of Conduct](https://github.com/Unidata/MetPy/blob/master/CODE_OF_CONDUCT.md).

Contributing
------------
**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not ready to be
an open source contributor; that your skills aren't nearly good enough to contribute. What
could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at all,
you can contribute code to open source. Contributing to open source projects is a fantastic
way to advance one's coding skills. Writing perfect code isn't the measure of a good developer
(that would disqualify all of us!); it's trying to create something, making mistakes, and
learning from those mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can help out by
writing documentation, tests, or even giving feedback about the project (and yes - that
includes giving feedback about the contribution process). Some of these contributions may be
the most valuable to the project as a whole, because you're coming to the project with fresh
eyes, so you can see the errors and assumptions that seasoned contributors have glossed over.

For more information, please read the see the [contributing guide](https://github.com/Unidata/MetPy/blob/master/CONTRIBUTING.md).

Philosophy
----------
The space MetPy aims for is GEMPAK (and maybe NCL)-like functionality, in a way that plugs
easily into the existing scientific Python ecosystem (numpy, scipy, matplotlib). So, if you
take the average GEMPAK script for a weather map, you need to:

- read data
- calculate a derived field
- show on a map/skew-T

One of the benefits hoped to achieve over GEMPAK is to make it easier to use these routines for
any meteorological Python application; this means making it easy to pull out the LCL
calculation and just use that, or re-use the Skew-T with your own data code. MetPy also prides
itself on being well-documented and well-tested, so that on-going maintenance is easily
manageable.

The intended audience is that of GEMPAK: researchers, educators, and any one wanting to script
up weather analysis. It doesn't even have to be scripting; all python meteorology tools are
hoped to be able to benefit from MetPy. Conversely, it's hoped to be the meteorological
equivalent of the audience of scipy/scikit-learn/skimage.
