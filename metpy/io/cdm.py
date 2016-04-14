# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
r'''The Common Data Model (CDM) is a data model for representing a wide array of data. The
goal is to be a simple, universal interface to different datasets. This API is a Python
implementation in the spirit of the original Java interface in netCDF-Java.
'''

from collections import OrderedDict

import numpy as np


class AttributeContainer(object):
    r"""A class to handle maintaining a list of netCDF attributes. Implements the attribute
    handling for other CDM classes."""
    def __init__(self):
        r"""Initialize an :class:`AttributeContainer`."""
        self._attrs = []

    def ncattrs(self):
        r"""Get a list of the names of the netCDF attributes.

        Returns
        -------
        List[str]
        """

        return self._attrs

    def __setattr__(self, key, value):
        if hasattr(self, '_attrs'):
            self._attrs.append(key)
        self.__dict__[key] = value

    def __delattr__(self, item):
        self.__dict__.pop(item)
        if hasattr(self, '_attrs'):
            self._attrs.remove(item)


class Group(AttributeContainer):
    r"""A Group holds dimensions and variables. Every CDM dataset has at least a root group.
    """
    def __init__(self, parent, name):
        r"""Initialize this :class:`Group`. Instead of constructing a :class:`Group` directly,
        you should use :meth:`~Group.createGroup`.

        Parameters
        ----------
        parent : Group or None
            The parent Group for this one. Passing in :data:`None` implies that this is
            the root :class:`Group`.
        name : str
            The name of this group

        See Also
        --------
        Group.createGroup
        """
        self.parent = parent
        if parent:
            self.parent.groups[name] = self

        #: :desc: The name of the :class:`Group`
        #: :type: str
        self.name = name

        #: :desc: Any Groups nested within this one
        #: :type: dict[str, Group]
        self.groups = OrderedDict()

        #: :desc: Variables contained within this group
        #: :type: dict[str, Variable]
        self.variables = OrderedDict()

        #: :desc: Dimensions contained within this group
        #: :type: dict[str, Dimension]
        self.dimensions = OrderedDict()

        # Do this last so earlier attributes aren't captured
        super(Group, self).__init__()

    # CamelCase API names for netcdf4-python compatibility
    def createGroup(self, name):  # noqa
        """Create a new Group as a descendant of this one.

        Parameters
        ----------
        name : str
            The name of the new Group.

        Returns
        -------
        Group
            The newly created :class:`Group`
        """
        grp = Group(self, name)
        self.groups[name] = grp
        return grp

    def createDimension(self, name, size):  # noqa
        """Create a new :class:`Dimension` in this :class:`Group`.

        Parameters
        ----------
        name : str
            The name of the new Dimension.
        size : int
            The size of the Dimension

        Returns
        -------
        Dimension
            The newly created :class:`Dimension`
        """
        dim = Dimension(self, name, size)
        self.dimensions[name] = dim
        return dim

    def createVariable(self, name, datatype, dimensions=(), fill_value=None, wrap_array=None):  # noqa
        """Create a new Variable in this Group.

        Parameters
        ----------
        name : str
            The name of the new Variable.
        datatype : str or numpy.dtype
            A valid Numpy dtype that describes the layout of the data within the Variable.
        dimensions : tuple[str], optional
            The dimensions of this Variable. Defaults to empty, which implies a scalar
            variable.
        fill_value : number, optional
            A scalar value that is used to fill the created storage. Defaults to None, which
            performs no filling, leaving the storage uninitialized.
        wrap_array : numpy.ndarray, optional
            Instead of creating an array, the Variable instance will assume ownership of the
            passed in array as its data storage. This is a performance optimization to avoid
            copying large data blocks. Defaults to None, which means a new array will be
            created.

        Returns
        -------
        Variable
            The newly created :class:`Variable`
        """
        var = Variable(self, name, datatype, dimensions, fill_value, wrap_array)
        self.variables[name] = var
        return var

    def __str__(self):
        print_groups = []
        if self.name:
            print_groups.append(self.name)

        if self.groups:
            print_groups.append('Groups:')
            for group in self.groups.values():
                print_groups.append(str(group))

        if self.dimensions:
            print_groups.append('\nDimensions:')
            for dim in self.dimensions.values():
                print_groups.append(str(dim))

        if self.variables:
            print_groups.append('\nVariables:')
            for var in self.variables.values():
                print_groups.append(str(var))

        if self.ncattrs():
            print_groups.append('\nAttributes:')
            for att in self.ncattrs():
                print_groups.append('\t{0}: {1}'.format(att, getattr(self, att)))
        return '\n'.join(print_groups)


class Dataset(Group):
    r"""A Dataset represents a set of data using the Common Data Model (CDM).

    This is currently only a wrapper around the root Group.
    """
    def __init__(self):
        """Initialize a Dataset."""
        super(Dataset, self).__init__(None, 'root')


class Variable(AttributeContainer):
    r"""A Variable holds typed data (using a :class:`numpy.ndarray`), as well as any relevant
    attributes (e.g. units).

    In addition to its various attributes, the Variable supports getting *and* setting data
    using the ``[]`` operator and indices or slices. Getting data returns
    :class:`numpy.ndarray` instances.
    """
    def __init__(self, group, name, datatype, dimensions, fill_value, wrap_array):
        """Initialize a Variable. Instead of constructing a Variable directly, you should
        use :meth:`Group.createVariable`.

        Parameters
        ----------
        group : Group
            The parent :class:`Group` that owns this Variable.
        name : str
            The name of this Variable.
        datatype : str or numpy.dtype
            A valid Numpy dtype that describes the layout of each element of the data
        dimensions : tuple[str], optional
            The dimensions of this Variable. Defaults to empty, which implies a scalar
            variable.
        fill_value : scalar, optional
            A scalar value that is used to fill the created storage. Defaults to None, which
            performs no filling, leaving the storage uninitialized.
        wrap_array : numpy.ndarray, optional
            Instead of creating an array, the Variable instance will assume ownership of the
            passed in array as its data storage. This is a performance optimization to avoid
            copying large data blocks. Defaults to None, which means a new array will be
            created.

        See Also
        --------
        Group.createVariable
        """
        # Initialize internal vars
        self._group = group
        self._name = name
        self._dimensions = tuple(dimensions)

        # Set the storage--create/wrap as necessary
        shape = tuple(len(group.dimensions.get(d)) for d in dimensions)
        if wrap_array is not None:
            if shape != wrap_array.shape:
                raise ValueError('Array to wrap does not match dimensions.')
            self._data = wrap_array
        else:
            self._data = np.empty(shape, dtype=datatype)
            if fill_value is not None:
                self._data.fill(fill_value)

        # Do this last so earlier attributes aren't captured
        super(Variable, self).__init__()

    # Not a property to maintain compatibility with NetCDF4 python
    def group(self):
        """Get the Group that owns this Variable.

        Returns
        -------
        Group
            The parent Group.
        """
        return self._group

    @property
    def name(self):
        'str: the name of the variable'
        return self._name

    @property
    def size(self):
        'int: the total number of elements'
        return self._data.size

    @property
    def shape(self):
        """tuple[int]: a tuple of integers describing the size of the Variable along each
        of its dimensions"""
        return self._data.shape

    @property
    def ndim(self):
        'int: the number of dimensions used by this variable'
        return self._data.ndim

    @property
    def dtype(self):
        """numpy.dtype: a valid Numpy :class:`~numpy.dtype` that describes the layout of each
        element of the data"""
        return self._data.dtype

    @property
    def datatype(self):
        """numpy.dtype: a valid Numpy :class:`~numpy.dtype` that describes the layout of each
        element of the data"""
        return self._data.dtype

    @property
    def dimensions(self):
        'tuple[str]: all the names of :class:`Dimension` used by this :class:`Variable`'
        return self._dimensions

    def __setitem__(self, ind, value):
        self._data[ind] = value

    def __getitem__(self, ind):
        return self._data[ind]

    def __str__(self):
        groups = [str(type(self)) +
                  ': {0.datatype} {0.name}({1})'.format(self, ', '.join(self.dimensions))]
        for att in self.ncattrs():
            groups.append('\t{0}: {1}'.format(att, getattr(self, att)))
        if self.ndim:
            if self.ndim > 1:
                shape_str = str(self.shape)
            else:
                shape_str = str(self.shape[0])
            groups.append('\tshape = ' + shape_str)
        return '\n'.join(groups)


# Punting on unlimited dimensions for now since we're relying upon numpy for storage
# We don't intend to be a full file API or anything, just need to be able to represent
# other files using a common API.
class Dimension(object):
    r"""A Dimension is used to represent a shared dimension between different Variables.
    For instance, variables that are dependent upon a common set of times.
    """
    def __init__(self, group, name, size=None):
        """Initialize a Dimension. Instead of constructing a Dimension directly, you should
        use ``Group.createDimension``.

        Parameters
        ----------
        group : Group
            The parent Group that owns this Variable.
        name : str
            The name of this Variable.
        size : int or None, optional
            The size of the Dimension. Defaults to None, which implies an empty dimension.

        See Also
        --------
        Group.createDimension
        """
        self._group = group

        #: :desc: The name of the Dimension
        #: :type: str
        self.name = name

        #: :desc: The size of this Dimension
        #: :type: int
        self.size = size

    # Not a property to maintain compatibility with NetCDF4 python
    def group(self):
        """Get the Group that owns this Dimension.

        Returns
        -------
        Group
            The parent Group.
        """
        return self._group

    def __len__(self):
        return self.size

    def __str__(self):
        return '{0}: name = {1.name}, size = {1.size}'.format(type(self), self)


# Not sure if this lives long-term or not
def cf_to_proj(var):
    r'''Converts a Variable with projection information conforming to the Climate and
    Forecasting (CF) netCDF conventions to a Proj.4 Projection instance.

    Parameters
    ----------
    var : Variable
        The projection variable with appropriate attributes.
    '''
    import pyproj
    kwargs = dict(lat_0=var.latitude_of_projection_origin,
                  a=var.earth_radius, b=var.earth_radius)
    if var.grid_mapping_name == 'lambert_conformal_conic':
        kwargs['proj'] = 'lcc'
        kwargs['lon_0'] = var.longitude_of_central_meridian
        kwargs['lat_1'] = var.standard_parallel
        kwargs['lat_2'] = var.standard_parallel
    if var.grid_mapping_name == 'polar_stereographic':
        kwargs['proj'] = 'stere'
        kwargs['lon_0'] = var.longitude_of_projection_origin
        kwargs['k_0'] = 1.0    # scale factor at natural origin
        kwargs['x_0'] = False  # Easting
        kwargs['y_0'] = False  # Northing

    return pyproj.Proj(**kwargs)
