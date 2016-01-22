# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from metpy.io.cdm import Dataset


def test_group():
    r'Test Group/Dataset behavior'
    ds = Dataset()
    ds.createDimension('x', 5)
    ds.createVariable('data', 'f4', ('x',), 5)
    ds.conventions = 'CF-1.5'

    assert 'x' in ds.dimensions
    assert 'data' in ds.variables
    assert 'conventions' in ds.ncattrs()

    assert str(ds) == ("root\n\nDimensions:\n"
                       "<class 'metpy.io.cdm.Dimension'>: name = x, size = 5\n\n"
                       "Variables:\n<class 'metpy.io.cdm.Variable'>: float32 data(x)\n\t"
                       "shape = 5\n\nAttributes:\n\tconventions: CF-1.5")


def test_dim():
    r'Test Dimension behavior'
    ds = Dataset()
    dim = ds.createDimension('x', 5)
    assert dim.size == 5
    assert dim.group() is ds
    assert str(dim) == "<class 'metpy.io.cdm.Dimension'>: name = x, size = 5"


def test_var():
    r'Test Variable behavior'
    ds = Dataset()
    ds.createDimension('x', 2)
    var = ds.createVariable('data', 'f4', ('x',), 5)

    assert 'data' in ds.variables
    assert var.shape == (2,)
    assert var.size == 2
    assert var.ndim == 1
    assert var.dtype == np.float32
    assert var[0] == 5

    var.units = 'meters'

    assert 'units' in var.ncattrs()
    assert var.units == 'meters'

    assert var.group() is ds

    assert str(var) == ("<class 'metpy.io.cdm.Variable'>: float32 data(x)"
                        "\n\tunits: meters\n\tshape = 2")


def test_multidim_var():
    r'Test multi-dim Variable'
    ds = Dataset()
    ds.createDimension('x', 2)
    ds.createDimension('y', 3)
    var = ds.createVariable('data', 'i8', ('x', 'y'))

    assert var.shape == (2, 3)
    assert var.size == 6
    assert var.ndim == 2
    assert var.dtype == np.int64

    assert str(var) == ("<class 'metpy.io.cdm.Variable'>: int64 data(x, y)"
                        "\n\tshape = (2, 3)")


def test_remove_attr():
    r'Test removing an attribute'
    ds = Dataset()
    ds.maker = 'me'
    assert 'maker' in ds.ncattrs()

    del ds.maker
    assert not hasattr(ds, 'maker')
    assert 'maker' not in ds.ncattrs()


def test_add_group():
    r'Test adding a group'
    ds = Dataset()
    grp = ds.createGroup('myGroup')
    assert grp.name == 'myGroup'
    assert 'myGroup' in ds.groups

    assert str(ds) == "root\nGroups:\nmyGroup"


def test_variable_size_check():
    r'Test Variable checking size of passed array'
    ds = Dataset()
    xdim = ds.createDimension('x', 2)
    ydim = ds.createDimension('y', 3)

    # Create array with dims flipped
    arr = np.empty((ydim.size, xdim.size), dtype='f4')

    with pytest.raises(ValueError):
        ds.createVariable('data', 'f4', ('x', 'y'), wrap_array=arr)
