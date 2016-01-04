# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from metpy.io.cdm import Dataset


class TestCDM(object):
    r'Tests for the CDM interface'

    @staticmethod
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

    @staticmethod
    def test_dim():
        r'Test Dimension behavior'
        ds = Dataset()
        dim = ds.createDimension('x', 5)
        assert dim.size == 5
        assert str(dim) == "<class 'metpy.io.cdm.Dimension'>: name = x, size = 5"

    @staticmethod
    def test_var():
        r'Test Variable behavior'
        ds = Dataset()
        ds.createDimension('x', 2)
        var = ds.createVariable('data', 'f4', ('x',), 5)

        assert 'data' in ds.variables
        assert var.shape == (2,)
        assert var.size == 2
        assert var[0] == 5

        var.units = 'meters'

        assert 'units' in var.ncattrs()
        assert var.units == 'meters'

        assert str(var) == ("<class 'metpy.io.cdm.Variable'>: float32 data(x)"
                            "\n\tunits: meters\n\tshape = 2")
