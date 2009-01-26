import numpy as np
from metpy.cbook import append_fields, add_dtype_titles, get_title

__all__ = ['Data', 'ArrayData']

class Data(object):
    _reserved = ['units', 'metadata']
    def __init__(self, data, units=None, metadata=None, descriptions=None):
        if units is None:
            units = dict()
        self.units = units

        if metadata is None:
            metadata = dict()
        self.metadata = metadata

        if descriptions is None:
            descriptions = dict()
        self._descriptions = descriptions

        self._inv_lookup = dict((v,k) for k,v in descriptions.iteritems())

        self._data = data

    def __setitem__(self, key, value):
        if key in self._reserved:
            self.__dict__[key] = value
        else:
            self._data[key] = value

    def __getitem__(self, key):
        if key in self._reserved:
            return self.__dict__[key]
        elif key in self._data:
            return self._data[key]
        else:
            return self._data.get(self._inv_lookup.get(key, None), None)

    def append_fields(self, names, arr, dtypes=None, units=None,
        descriptions=None):
        if dtypes is None:
            dtypes = [a.dtype for a in arr]

        for n,a,dt in zip(names, arr, dtypes):
            self._data[n] = a.astype(dt)
        self.units.update(zip(names, units))
        self.descriptions.update(zip(names, descriptions))

    def get_descrip(self, name):
        return self._descriptions.get(name, '')

    def set_descrip(self, name, desc):
        self._descriptions[name] = desc
        self._inv_lookup[desc] = name

class ArrayData(Data):
    def __init__(self, data, units=None, metadata=None, descriptions=None):
        Data.__init__(self, data, units, metadata, descriptions)
        if self._descriptions:
            add_dtype_titles(self._data, descriptions)
            self._descriptions = None

    def __setitem__(self, key, value):
        if key in self._reserved:
            self.__dict__[key] = value
        else:
            self._data[key] = value

    def __getitem__(self, key):
        if key in self._reserved:
            return self.__dict__[key]
        else:
            return self._data[key]

    def append_fields(self, names, arr, dtypes=None, units=None,
        descriptions=None):
        # This makes the names now a list of tuples of names and titles,
        # so that the dtype constructor can use them
        self.units.update(zip(names, units))

        names = zip(descriptions, names)
        self._data = append_fields(self._data, names, arr, dtypes)

    def get_descrip(self, name):
        return get_title(self._data, name)

    def set_descrip(self, name, desc):
        self._data = add_dtype_titles(self._data, {name:desc})
