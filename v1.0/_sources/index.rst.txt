.. image:: _static/sounding.png
   :width: 150 px
   :align: left

.. image:: _static/radar.png
   :width: 150 px
   :align: left

.. image:: _static/NSF.jpg
   :width: 100 px
   :align: right

.. toctree::
   :maxdepth: 1
   :hidden:

   installguide
   startingguide
   units
   examples/index
   Python Gallery (separate site) <https://unidata.github.io/python-training/gallery/gallery-home>
   tutorials/index
   api/index
   roadmap
   gempak
   SUPPORT
   CONTRIBUTING
   infrastructureguide
   citing
   references

=====
MetPy
=====

MetPy is a collection of tools in Python for reading, visualizing, and
performing calculations with weather data. If you're new to MetPy, check
out our :doc:`Getting Started <startingguide>` guide. Development is
supported by the National Science Foundation through grants AGS-1344155,
OAC-1740315, AGS-1901712.

For additional MetPy examples not included in this repository, please see the `Unidata Python
Gallery`_.

We support Python >= 3.6. Support for Python 2.7 was dropped with the release of 0.12.

----------
Contact Us
----------

* For questions about MetPy, please ask them using the "metpy" tag on StackOverflow_. Our
  developers are actively monitoring for questions there.
* You can also email `Unidata's
  python support email address <mailto: support-python@unidata.ucar.edu>`_
* The source code is available on GitHub_
* Bug reports and feature requests should be directed to the
  `GitHub issue tracker`__
* MetPy has a Gitter_ chatroom for more "live" communication
* MetPy can also be found on Twitter_
* If you use MetPy in a publication, please see :ref:`Citing_MetPy`.
* For release announcements, join Unidata's python-users_ mailing list

.. _python-users: https://www.unidata.ucar.edu/support/#mailinglists
.. _GitHub: https://github.com/Unidata/MetPy
.. _Gitter: https://gitter.im/Unidata/MetPy
.. _Twitter: https://twitter.com/MetPy
.. _StackOverflow: https://stackoverflow.com/questions/tagged/metpy
__ https://github.com/Unidata/MetPy/issues

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

-----
Media
-----

* `AMS 2020 talk`_ on MetPy being ready for a 1.0 release
* `AMS 2019 talk`_ on bringing GEMPAK-like syntax to MetPy's declaritive plotting interface
* `AMS 2019 poster`_ on recent development and community building with MetPy
* `SciPy 2018 poster`_ and `abstract <http://johnrleeman.com/pubs/2018/Leeman_2018_SciPy_Abstract.pdf>`_ on building community by John Leeman
* `SciPy 2018 talk`_ on prototyping MetPy's future declarative plotting interface
* Presentation on MetPy and Community Development at the `2018 AMS Annual Meeting`_ by Ryan May
* `SciPy 2017 poster`_ and `repository <https://github.com/jrleeman/CAPE-SciPy-2017>`_
  about reproducing a classic CAPE paper with MetPy.
* `SciPy 2017 talk`_ and `slides
  <https://nbviewer.jupyter.org/format/slides/github/dopplershift/
  Talks/blob/master/SciPy2017/MetPy%20Units.ipynb>`_
  about challenges developing MetPy with units
* MetPy was featured on `Episode 100 of Podcast.__init__`_
* Presentation on MetPy's build infrastructure by Ryan May at `SciPy 2016`_
* MetPy was included in tools presented at the `SSEC/Wisconsin AOS Python Workshop`_
* Presentation on MetPy at the `2016 AMS Annual Meeting`_ by Ryan May
* Ryan May's talk and tutorial on MetPy at the `2015 Unidata Users Workshop`_

.. _`2015 Unidata Users Workshop`: https://www.youtube.com/watch?v=umwauHAL-0M
.. _`2016 AMS Annual Meeting`: https://ams.confex.com/ams/96Annual/webprogram/Paper286983.html
.. _`SSEC/Wisconsin AOS Python Workshop`: https://www.youtube.com/watch?v=RRvJI_vouQc
.. _`SciPy 2016`: https://www.youtube.com/watch?v=moLKGjbXvgE
.. _`Episode 100 of Podcast.__init__`: https://www.podcastinit.com/episode-100-metpy-with-ryan-may-sean-arms-and-john-leeman/
.. _`SciPy 2017 talk`: https://www.youtube.com/watch?v=qCo9bkT9sow
.. _`SciPy 2017 poster`: https://github.com/jrleeman/CAPE-SciPy-2017/blob/master/Poster/SciPy_Poster_2017.pdf
.. _`2018 AMS Annual Meeting`: https://ams.confex.com/ams/98Annual/webprogram/Paper333578.html
.. _`SciPy 2018 talk`: https://www.youtube.com/watch?v=OKQlUdPY0Jc
.. _`SciPy 2018 poster`: http://johnrleeman.com/pubs/2018/Leeman_2018_SciPy_Poster.pdf
.. _`AMS 2019 talk`: https://ams.confex.com/ams/2019Annual/meetingapp.cgi/Paper/352384
.. _`AMS 2019 poster`: https://ams.confex.com/ams/2019Annual/meetingapp.cgi/Paper/354058
.. _`AMS 2020 talk`: https://ams.confex.com/ams/2020Annual/meetingapp.cgi/Paper/369011

-------
License
-------

MetPy is available under the terms of the open source `BSD 3 Clause license`__.

__ https://raw.githubusercontent.com/Unidata/MetPy/master/LICENSE

---------------
Code of Conduct
---------------
We want everyone to feel welcome to contribute to MetPy and participate in discussions. In that
spirit please have a look at our `code of conduct`__.

__ https://github.com/Unidata/MetPy/blob/master/CODE_OF_CONDUCT.md

----------------
Related Projects
----------------

* netCDF4-python_ is the officially blessed Python API for netCDF_
* siphon_ is a Python API for accessing remote data on `THREDDS Data Servers`__
* The `Unidata Python Gallery`_ is a collection of meteorological Python scripts

.. _netCDF4-python: https://unidata.github.io/netcdf4-python/
.. _netCDF: https://www.unidata.ucar.edu/software/netcdf/
.. _siphon: https://unidata.github.io/siphon/
.. _Unidata Python Gallery: https://unidata.github.io/python-training/gallery/gallery-home
__ https://www.unidata.ucar.edu/software/tds/current/
