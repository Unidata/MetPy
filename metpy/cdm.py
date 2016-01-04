# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from collections import OrderedDict

import numpy as np


class AttributeContainer(object):
    def __init__(self):
        self._attrs = []

    def ncattrs(self):
        """Get a list of the names of the netCDF attributes.

        Returns
        -------
        list(str)
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
    def __init__(self, parent, name):
        self.parent = parent
        if parent:
            self.parent.groups[name] = self

        self.name = name
        self.groups = OrderedDict()
        self.variables = OrderedDict()
        self.dimensions = OrderedDict()

        # Do this last so earlier attributes aren't captured
        super(Group, self).__init__()

    # CamelCase API names for netcdf4-python compatibility
    def createGroup(self, name):  # noqa
        grp = Group(self, name)
        self.groups[name] = grp
        return grp

    def createDimension(self, name, size):  # noqa
        dim = Dimension(self, name, size)
        self.dimensions[name] = dim
        return dim

    def createVariable(self, name, datatype, dimensions=(), fill_value=None, wrap_array=None):  # noqa
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
    def __init__(self):
        super(Dataset, self).__init__(None, 'root')


class Variable(AttributeContainer):
    def __init__(self, group, name, datatype, dimensions, fill_value, wrap_array):
        # Initialize internal vars
        self._group = group
        self._name = name
        group.variables[name] = self
        self._dimensions = dimensions

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
        return self._group

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._data.size

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def datatype(self):
        return self._data.dtype

    @property
    def dimensions(self):
        return self._dimensions

    def _init_storage(self):
        self._data = None

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
    def __init__(self, group, name, size=None):
        self._group = group
        self.name = name
        self.size = size

    # Not a property to maintain compatibility with NetCDF4 python
    def group(self):
        return self._group

    def __len__(self):
        return self.size

    def __str__(self):
        return '{0} name = {1.name}, size = {1.size}'.format(type(self), self.name)


# Not sure if this lives long-term or not
def cf_to_proj(var):
    import pyproj
    kwargs = dict(lat_0=var.latitude_of_projection_origin,
                  lon_0=var.longitude_of_central_meridian,
                  a=var.earth_radius, b=var.earth_radius)
    if var.grid_mapping_name == 'lambert_conformal_conic':
        kwargs['proj'] = 'lcc'
        kwargs['lat_1'] = var.standard_parallel
        kwargs['lat_2'] = var.standard_parallel

    return pyproj.Proj(**kwargs)
