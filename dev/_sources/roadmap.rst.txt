=============
MetPy Roadmap
=============

----
Goal
----

MetPy’s goal is to provide the domain-specific tools for building meteorological and
atmospheric science applications in Python. This means supporting scripted workflows, like
those done previously using GEMPAK, NCL, etc. MetPy’s functionality is not limited to
supporting only static views, we also envision being used as the calculation core for graphical
applications as well. Our focus on providing the domain-specific tools means that general
functionality is moved upstream (to e.g. matplotlib, xarray) whenever sensible and possible.

-----
Plans
-----

We enumerate here general plans for how we plan to advance MetPy over the next couple of years.
We have intentionally avoided assigning dates to any of these, acknowledging our complete
inability to estimate time and level of effort, as well as because our priorities are flexible
and are informed by community feedback. The order of the following items reflects a rough
prioritized order, but is by no means a strict ordering.

Our plan is also to release MetPy 1.0 in 2019. Items in the roadmap relevant to the 1.0 release
are specifically called out below.

We welcome input and discussion about these items over on MetPy’s
`issue tracker <https://github.com/Unidata/MetPy/issues>`_.

For a more detailed view of plans of specific issues and pull requests for upcoming versions
of MetPy, visit the `GitHub milestones page <https://github.com/Unidata/MetPy/milestones>`_.
These represent our best idea of what's planned for a specific upcoming version, though are
subject to rapid change as release dates approach.

~~~~~~~~~~~~~~~~~~~~
Declarative Plotting
~~~~~~~~~~~~~~~~~~~~

One of GEMPAK’s successes is the simplicity with which plots can be created. Part of this
simplicity is enforced by the interface to GEMPAK: setting variables. MetPy 0.10 introduced the
simplified plotting interface to begin to replicate this, with initial support for image and
contour plots. We plan to continue to advance this interface with additional plot types and
other features:

* Vector plots
* Streamlines
* Filled contours
* Polar Radar Data
* Cross-sections
* Skew-T
* Plot decorations: logos, timestamp, colorbars, labelling contours, figure title

Replicating all of matplotlib’s features is beyond the scope of this interface. We intend,
instead, to aim for simplicity of the interface while replicating GEMPAK’s capabilities and
being able to do what is needed for most common use cases.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Xarray Integration and Unit Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This item is the most important to having stable interfaces in advance of a 1.0 release of
MetPy. Xarray has quickly proven to be a capable data model on which to base MetPy’s handling
of gridded data (tabular/point data will likely use pandas). MetPy’s unit-handling based on
pint has been a cornerstone for providing robust calculations that are difficult to use (from a
dimensionality perspective) incorrectly. It is unfortunate, then, that the unit handling has
also proven to be MetPy’s most significant learning hurdle for new users, as well as the
biggest technical challenge to broader adoption of xarray within MetPy’s code base. We have
identified a few tasks that need to be done in this area to improve things:

* Use xarray natively for all calculation interfaces; unit information can be pulled from
  metadata. This allows calculations to be coordinate-aware, which is a huge benefit for so
  many calculations (e.g. vorticity, isentropic interpolation)
* Continue to accept numpy array + pint, but convert to xarray internally
* Add helper functions to simplify xarray creation and manipulation for common use cases
* Simplify manipulation and display of dates from xarray (may go upstream)
* Investigate other unit library solutions (e.g. unyt) for better xarray integration

The first two items reverse the current state of MetPy’s calculations, which is to take xarray,
but convert to numpy + pint; the benefit is that unit information can be reconstructed as
necessary from xarray, but all calculations will then have access to coordinate information.
This is a 1.0 item because many calculations involving coordinate information will likely
change to simplify their interfaces.

~~~~~~~~~~~~
Calculations
~~~~~~~~~~~~

In addition to the work at the calculation infrastructure level, we have identified some
specific calculations that are necessary to consider MetPy a solution to the majority of the
problems users previously solved with GEMPAK:

* Indices (e.g. SWEAT, K-Index); these require the improved xarray integration to simplify
  access across (usually vertical) coordinates
* Dynamic tropopause
* Elliptic PDE solver (leveraging SciPy)
* Richardson number
* Rossby number
* Flux divergence (leading to e.g. Ageopotential flux)

We also recognize a need to continue to improve the robustness of many of the sounding-based
calculations. These calculations tend to fail in many challenging real-world data cases.

One major area of development for the calculations is what we refer to as the “automated
solver.” The goal of the solver is to automate calculation of quantities of interest from
datasets. This reduces the need for variations on calculation functions that only change based
on the type of input data available. For example, this eliminates the need for different ways
to calculate relative humidity based on the available moisture parameters. This solver would
also eliminate the need to identify complicated sequences of function calls to go from standard
model output to fields of interest. This solver is enabled by xarray’s concept of Datasets as
well as by leveraging the netCDF Climate and Forecasting metadata standard’s concept of
“standard names” for identifying the types of variables. The standard function interface will
still exist for challenging cases. The solver will also enable the plotting of derived
quantities through the simplified plotting interface.

~~~~~~~~~~~~~~~~~~~
File Format Support
~~~~~~~~~~~~~~~~~~~

MetPy’s file format support is an important part of its feature set, and we want to continue to
expand the set of formats to which MetPy facilitates access. This also fits with filling
GEMPAK’s feature set, since its support for a variety of data formats was important to its
utility. Unlike GEMPAK, MetPy will never rely on a set of “decoders” to translate datafiles to
a new format. This goes against the spirit of fitting within the rest of the scientific Python
ecosystem. Instead, MetPy will provide tools that read data files and provide the data in one
of two data structures: Xarray Dataset or Pandas DataFrame. These two data structures are
widely used within the broader ecosystem.

We currently have identified the following formats as priority for support (in rough
prioritized order):

* BUFR
* METAR
* MCIDAS Area Files
* GRIB

We feel these formats are important to have Python support in order to ensure that Python users
have ready support for common, real-time data sources. GRIB currently is down on this list not
due to a lack of importance, but because the cfgrib and eccodes projects currently cover this,
in terms of access to data through xarray. In the future, though, the MetPy developers would
like to explore providing a pure-python solution for GRIB.

We would also like to support a wide variety of information and observations that the U.S.
National Weather Service distributes through text bulletins (e.g. hurricane dropsondes,
aircraft reconnaissance, frontal positions). Support is planned, but considered secondary to
those formats above.

~~~~~~~~~~~
Other Items
~~~~~~~~~~~

Here are a few more items that did not fit above:

* Performance optimization

  - Moving calculations to Numba, Cython, etc. Numba would be the preferred solution, because
    it would not incur packaging challenges. Past experiments with Numba and MetPy have not
    been promising, though.
  - Make more calculations (e.g. CAPE) work on grids of data

* Drop Python 2! For more info see :ref:`python27`.
