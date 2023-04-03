# Copyright (c) 2015,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r"""Module to provide unit support.

This makes use of the ``pint`` library and sets up the default settings
for good temperature support.

See Also: :doc:`Working with Units </tutorials/unit_tutorial>`.

Attributes
----------
units : :class: `pint.UnitRegistry()`
    The unit registry used throughout the package. Any use of units in MetPy should
    import this registry and use it to grab units.

"""
import contextlib
import functools
from inspect import Parameter, signature
import logging
import re
import warnings

import numpy as np
import pint

log = logging.getLogger(__name__)

UndefinedUnitError = pint.UndefinedUnitError
DimensionalityError = pint.DimensionalityError

_base_unit_of_dimensionality = {
    '[pressure]': 'Pa',
    '[temperature]': 'K',
    '[dimensionless]': '',
    '[length]': 'm',
    '[speed]': 'm s**-1'
}


def _fix_udunits_powers(string):
    """Replace UDUNITS-style powers (m2 s-2) with exponent symbols (m**2 s**-2)."""
    return _UDUNIT_POWER.sub('**', string)


# Fix UDUNITS-style powers and percent signs
_UDUNIT_POWER = re.compile(r'(?<=[A-Za-z\)])(?![A-Za-z\)])'
                           r'(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])')
_unit_preprocessors = [_fix_udunits_powers, lambda string: string.replace('%', 'percent')]


def setup_registry(reg):
    """Set up a given registry with MetPy's default tweaks and settings."""
    reg.autoconvert_offset_to_baseunit = True

    # For Pint 0.18.0, need to deal with the fact that the wrapper isn't forwarding on setting
    # the attribute.
    with contextlib.suppress(AttributeError):
        reg.get().autoconvert_offset_to_baseunit = True

    for pre in _unit_preprocessors:
        if pre not in reg.preprocessors:
            reg.preprocessors.append(pre)

    # Add a percent unit if it's not already present, it was added in 0.21
    if 'percent' not in reg:
        reg.define('percent = 0.01 = %')

    # Define commonly encountered units not defined by pint
    reg.define('degrees_north = degree = degrees_N = degreesN = degree_north = degree_N '
               '= degreeN')
    reg.define('degrees_east = degree = degrees_E = degreesE = degree_east = degree_E '
               '= degreeE')

    # Alias geopotential meters (gpm) to just meters
    reg.define('@alias meter = gpm')

    # Enable pint's built-in matplotlib support
    reg.setup_matplotlib()

    return reg


# Make our modifications using pint's application registry--which allows us to better
# interoperate with other libraries using Pint.
units = setup_registry(pint.get_application_registry())

# Silence UnitStrippedWarning
warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)


def pandas_dataframe_to_unit_arrays(df, column_units=None):
    """Attach units to data in pandas dataframes and return quantities.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data in pandas dataframe.

    column_units : dict
        Dictionary of units to attach to columns of the dataframe. Overrides
        the units attribute if it is attached to the dataframe.

    Returns
    -------
        Dictionary containing `Quantity` instances with keys corresponding to the dataframe
        column names.

    """
    if not column_units:
        try:
            column_units = df.units
        except AttributeError:
            raise ValueError('No units attribute attached to pandas '
                             'dataframe and col_units not given.') from None

    # Iterate through columns attaching units if we have them, if not, don't touch it
    res = {}
    for column in df:
        if column in column_units and column_units[column]:
            res[column] = units.Quantity(df[column].values, column_units[column])
        else:
            res[column] = df[column].values
    return res


def is_quantity(*args):
    """Check whether an instance is a quantity."""
    return all(isinstance(a, pint.Quantity) for a in args)


def concatenate(arrs, axis=0):
    r"""Concatenate multiple values into a new quantity.

    This is essentially a scalar-/masked array-aware version of `numpy.concatenate`. All items
    must be able to be converted to the same units. If an item has no units, it will be given
    those of the rest of the collection, without conversion. The first units found in the
    arguments is used as the final output units.

    Parameters
    ----------
    arrs : Sequence[pint.Quantity or numpy.ndarray]
        The items to be joined together

    axis : int, optional
        The array axis along which to join the arrays. Defaults to 0 (the first dimension)

    Returns
    -------
    `pint.Quantity`
        New container with the value passed in and units corresponding to the first item.

    """
    dest = 'dimensionless'
    for a in arrs:
        if hasattr(a, 'units'):
            dest = a.units
            break

    data = []
    for a in arrs:
        if hasattr(a, 'to'):
            a = a.to(dest).magnitude
        data.append(np.atleast_1d(a))

    # Use masked array concatenate to ensure masks are preserved, but convert to an
    # array if there are no masked values.
    data = np.ma.concatenate(data, axis=axis)
    if not np.any(data.mask):
        data = np.asarray(data)

    return units.Quantity(data, dest)


def masked_array(data, data_units=None, **kwargs):
    """Create a :class:`numpy.ma.MaskedArray` with units attached.

    This is a thin wrapper around :class:`numpy.ma.MaskedArray` that ensures that
    units are properly attached to the result (otherwise units are silently lost). Units
    are taken from the ``data_units`` argument, or if this is ``None``, the units on ``data``
    are used.

    Parameters
    ----------
    data : array-like
        The source data. If ``data_units`` is `None`, this should be a `pint.Quantity` with
        the desired units.
    data_units : str or `pint.Unit`, optional
        The units for the resulting `pint.Quantity`
    kwargs
        Arbitrary keyword arguments passed to `numpy.ma.masked_array`, optional

    Returns
    -------
    `pint.Quantity`

    """
    if data_units is None:
        data_units = data.units
    return units.Quantity(np.ma.masked_array(data, **kwargs), data_units)


def _mutate_arguments(bound_args, check_type, mutate_arg):
    """Handle adjusting bound arguments.

    Calls ``mutate_arg`` on every argument, including those passed as ``*args``, if they are
    of type ``check_type``.
    """
    for arg_name, arg_val in bound_args.arguments.items():
        if isinstance(arg_val, check_type):
            bound_args.arguments[arg_name] = mutate_arg(arg_val, arg_name)

    if isinstance(bound_args.arguments.get('args'), tuple):
        bound_args.arguments['args'] = tuple(
            mutate_arg(arg_val, '(unnamed)') if isinstance(arg_val, check_type) else arg_val
            for arg_val in bound_args.arguments['args'])


def _check_argument_units(args, defaults, dimensionality):
    """Yield arguments with improper dimensionality."""
    for arg, val in args.items():
        # Get the needed dimensionality (for printing) as well as cached, parsed version
        # for this argument.
        try:
            need, parsed = dimensionality[arg]
        except KeyError:
            # Argument did not have units specified in decorator
            continue

        if arg in defaults and (defaults[arg] is not None or val is None):
            check = val == defaults[arg]
            if np.all(check):
                continue

        # See if the value passed in is appropriate
        try:
            if val.dimensionality != parsed:
                yield arg, val, val.units, need
        # No dimensionality
        except AttributeError:
            # If this argument is dimensionless, don't worry
            if parsed != '':
                yield arg, val, 'none', need


def _get_changed_version(docstring):
    """Find the most recent version in which the docs say a function changed."""
    matches = re.findall(r'.. versionchanged:: ([\d.]+)', docstring)
    return max(matches) if matches else None


def _check_units_outer_helper(func, *args, **kwargs):
    """Get dims and defaults from function signature and specified dimensionalities."""
    # Match the signature of the function to the arguments given to the decorator
    sig = signature(func)
    bound_units = sig.bind_partial(*args, **kwargs)

    # Convert our specified dimensionality (e.g. "[pressure]") to one used by
    # pint directly (e.g. "[mass] / [length] / [time]**2). This is for both efficiency
    # reasons and to ensure that problems with the decorator are caught at import,
    # rather than runtime.
    dims = {name: (orig, units.get_dimensionality(orig.replace('dimensionless', '')))
            for name, orig in bound_units.arguments.items()}

    defaults = {name: sig.parameters[name].default for name in sig.parameters
                if sig.parameters[name].default is not Parameter.empty}

    return sig, dims, defaults


def _check_units_inner_helper(func, sig, defaults, dims, *args, **kwargs):
    """Check bound arguments for unit correctness."""
    # Match all passed in value to their proper arguments so we can check units
    bound_args = sig.bind(*args, **kwargs)
    bad = list(_check_argument_units(bound_args.arguments, defaults, dims))

    # If there are any bad units, emit a proper error message making it clear
    # what went wrong.
    if bad:
        msg = f'`{func.__name__}` given arguments with incorrect units: '
        msg += ', '.join(
            f'`{arg}` requires "{req}" but given "{given}"' for arg, _, given, req in bad
        )
        if 'none' in msg:
            if any(isinstance(x, np.ma.core.MaskedArray) for _, x, _, _ in bad):
                msg += ('\nA masked array `m` can be assigned a unit as follows:\n'
                        '    from metpy.units import units\n'
                        '    m = units.Quantity(m, "m/s")')
            else:
                msg += ('\nA xarray DataArray or numpy array `x` can be assigned a unit as '
                        'follows:\n'
                        '    from metpy.units import units\n'
                        '    x = x * units("m/s")')
            msg += ('\nFor more information see the Units Tutorial: '
                    'https://unidata.github.io/MetPy/latest/tutorials/unit_tutorial.html')

        # If function has changed, mention that fact
        if func.__doc__:
            changed_version = _get_changed_version(func.__doc__)
            if changed_version:
                msg = (
                    f'This function changed in {changed_version}--double check '
                    'that the function is being called properly.\n'
                ) + msg
        raise ValueError(msg)

    # Return the bound arguments for reuse
    return bound_args


def check_units(*units_by_pos, **units_by_name):
    """Create a decorator to check units of function arguments."""
    def dec(func):
        sig, dims, defaults = _check_units_outer_helper(func, *units_by_pos, **units_by_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _check_units_inner_helper(func, sig, defaults, dims, *args, **kwargs)
            return func(*args, **kwargs)

        return wrapper
    return dec


def process_units(
    input_dimensionalities,
    output_dimensionalities,
    output_to=None,
    ignore_inputs_for_output=None
):
    """Wrap a non-Quantity-using function in base units to fully handle units."""
    def dec(func):
        sig, dims, defaults = _check_units_outer_helper(func, **input_dimensionalities)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = _check_units_inner_helper(func, sig, defaults, dims, *args, **kwargs)

            # Determine unit(s) with which to wrap output (first, since we mutate the bound
            # args)
            if isinstance(output_dimensionalities, tuple):
                multiple_output = True
                outputs = output_dimensionalities
            else:
                multiple_output = False
                outputs = (output_dimensionalities,)
            output_control = []
            for i, output in enumerate(outputs):
                convert_to = (
                    output_to if not multiple_output or output_to is None else output_to[i]
                )
                # Find matching input, if it exists
                if convert_to is None:
                    for name, (this_dim, _) in dims.items():
                        if (
                            this_dim == output
                            and (
                                ignore_inputs_for_output is None
                                or name not in ignore_inputs_for_output
                            )
                        ):
                            try:
                                convert_to = bound_args.arguments[name].units
                            except AttributeError:
                                # We don't have units, so given prior check, is dimensionless
                                convert_to = ''
                            break

                output_control.append((_base_unit_of_dimensionality[output], convert_to))

            # Convert all inputs as specified, assuming dimensionality is fine based on above
            _mutate_arguments(bound_args, units.Quantity, lambda val, _: val.to_base_units().m)

            # Evaluate inner calculation
            result = func(*bound_args.args, **bound_args.kwargs)

            # Wrap output
            if multiple_output:
                wrapped_result = []
                for this_result, this_output_control in zip(result, output_control):
                    q = units.Quantity(this_result, this_output_control[0])
                    if this_output_control[1] is not None:
                        q = q.to(this_output_control[1])
                    wrapped_result.append(q)
                return tuple(wrapped_result)
            else:
                q = units.Quantity(result, output_control[0][0])
                if output_control[0][1] is not None:
                    q = q.to(output_control[0][1])
                return q

        # Attach the unwrapped func for internal use
        wrapper._nounit = func

        return wrapper
    return dec
