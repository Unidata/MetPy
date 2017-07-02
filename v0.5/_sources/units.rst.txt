Unit Support
============

To ensure correct calculations, MetPy relies upon the :mod:`pint` library to enforce
unit-correctness. This simplifies the MetPy API by eliminating the need to specify units
various functions. Instead, only the final results need to be converted to desired units. For
more information on unit support, see the documentation for
`Pint <http://pint.readthedocs.io>`_. Particular attention should be paid to the support
for `temperature units <http://pint.readthedocs.io/en/latest/nonmult.html>`_.

------------
Construction
------------

To use units, the first step is to import the default MetPy units registry from the
:mod:`~metpy.units` module:

.. code-block:: python

    import numpy as np
    from metpy.units import units

The unit registry encapsulates all of the available units, as well as any pertinent settings.
The registry also understands unit prefixes and suffixes; this allows the registry to
understand ``'kilometer'`` and ``'meters'`` in addition to the base ``'meter'`` unit.

In general, using units is only a small step on top of using the :class:`numpy.ndarray` object.
The easiest way to attach units to an array is to multiply by the units:

.. code-block:: python

    distance = np.arange(1, 5) * units.meters

It is also possible to directly construct a :class:`pint.Quantity`, with a full units string:

.. code-block:: python

    time = units.Quantity(np.arange(2, 10, 2), 'sec')

Compound units can be constructed by the direct mathematical operations necessary:

.. code-block:: python

    g = 9.81 * units.meter / (units.second * units.second)

This verbose syntax can be reduced by using the unit registry's support for parsing units:

.. code-block:: python

    g = 9.81 * units('m/s^2')

----------
Operations
----------

With units attached, it is possible to perform mathematical operations, resulting in the proper
units:

.. code-block:: python

    print(distance / time)

.. parsed-literal::
    [ 0.5  0.5  0.5  0.5] meter / second

For multiplication and division, units can combine and cancel. For addition and subtraction,
instead the operands must have compatible units. For instance, this works:

.. code-block:: python

    print(distance + distance)

.. parsed-literal::

    [0 2 4 6 8] meter

But this does not:

.. code-block:: python

    print(distance + time)

.. parsed-literal::
    DimensionalityError: Cannot convert from 'meter' ([length]) to 'second' ([time])

Even if the units are not identical, as long as they are dimensionally equivalent, the
operation can be performed:

.. code-block:: python

    print(3 * units.inch + 5 * units.cm)

.. parsed-literal::
    4.968503937007874 inch

----------
Conversion
----------

Converting a :class:`~pint.Quantity` between units can be accomplished by using the
:meth:`~pint.Quantity.to` method call, which constructs a new :class:`~pint.Quantity` in the
desired units:

.. code-block:: python

    print((1 * units.inch).to(units.mm))

.. parsed-literal::
    25.400000000000002 millimeter

There is also the :meth:`~pint.Quantity.ito` method which performs the same operation in place.
To simplify units, there is also the :meth:`~pint.Quantity.to_base_units` method, which
converts a quantity to SI units, performing any needed cancellation:

.. code-block:: python

    Lf = 3.34e6 * units('J/kg')
    print(Lf, Lf.to_base_units(), sep='\n')

.. parsed-literal::
    3340000.0 joule / kilogram
    3340000.0 meter ** 2 / second ** 2

:meth:`~pint.Quantity.to_base_units` can also be done in place via the
:meth:`~pint.Quantity.ito_base_units` method.
