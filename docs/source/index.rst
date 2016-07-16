.. MetPy documentation master file, created by
   sphinx-quickstart on Wed Apr 22 15:27:44 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/sounding.png
   :width: 150 px
   :align: left

.. image:: _static/radar.png
   :width: 150 px
   :align: left

=====
MetPy
=====

MetPy is a collection of tools in Python for reading, visualizing, and
performing calculations with weather data.

MetPy is still in an early stage of development, and as such
**no APIs are considered stable.** While we won't break things
just for fun, many things may still change as we work through
design issues.

We support Python 2.7 as well as Python >= 3.3.

-------------
Documentation
-------------

.. toctree::
   :maxdepth: 1

   installguide
   units
   examples/index
   api/index
   developerguide

----------
Contact Us
----------

* For questions and discussion about MetPy, join Unidata's python-users_
  mailing list
* The source code is available on GitHub_
* Bug reports and feature requests should be directed to the
  `GitHub issue tracker`__
* MetPy has a Gitter_ chatroom for more "live" communication
* MetPy can also be found on Twitter_

.. _python-users: https://www.unidata.ucar.edu/support/#mailinglists
.. _GitHub: https://github.com/metpy/MetPy
.. _Gitter: https://gitter.im/metpy/MetPy
.. _Twitter: https://twitter.com/MetPy
__ https://github.com/metpy/MetPy/issues

-------------
Presentations
-------------

* Presentation on MetPy's build infrastructure by Ryan May at `SciPy 2016`_.
* MetPy was included in tools presented at the `SSEC/Wisconsin AOS Python Workshop`_.
* Presentation on MetPy at the `2016 AMS Annual Meeting`_ by Ryan May.
* Ryan May's talk and tutorial on MetPy at the `2015 Unidata Users Workshop`_.

.. _`2015 Unidata Users Workshop`: https://www.youtube.com/watch?v=umwauHAL-0M
.. _`2016 AMS Annual Meeting`: https://ams.confex.com/ams/96Annual/webprogram/Paper286983.html
.. _`SciPy 2016`: https://youtu.be/moLKGjbXvgE
.. _`SSEC/Wisconsin AOS Python Workshop`: https://youtu.be/RRvJI_vouQc

-------
License
-------

MetPy is available under the terms of the open source `BSD 3 Clause license`__.

__ https://raw.githubusercontent.com/metpy/MetPy/master/LICENSE

----------------
Related Projects
----------------

* netCDF4-python_ is the officially blessed Python API for netCDF_
* siphon_ is an API for accessing remote data on `THREDDS Data Server`__

.. _netCDF4-python: https://unidata.github.io/netcdf4-python/
.. _netCDF: https://www.unidata.ucar.edu/software/netcdf/
.. _siphon: http://siphon.readthedocs.org
__ https://www.unidata.ucar.edu/software/thredds/current/tds/

