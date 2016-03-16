# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import os.path
import tempfile
try:
    from StringIO import StringIO
    buffer_args = dict(bufsize=0)
except ImportError:
    from io import StringIO
    buffer_args = dict(buffering=1)

import numpy as np
from metpy.plots.ctables import ColortableRegistry, convert_gempak_table


class TestColortableRegistry(object):
    'Tests for ColortableRegistry'
    def setup_method(self, _):  # noqa
        'Set up a registry for use by the tests.'
        self.reg = ColortableRegistry()

    def test_package_resource(self):
        'Test registry scanning package resource'
        self.reg.scan_resource('metpy.plots', 'nexrad_tables')
        assert 'cc_table' in self.reg

    def test_scan_dir(self):
        'Test registry scanning a directory'
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir='.', suffix='.tbl', delete=False,
                                             **buffer_args) as fobj:
                fobj.write('"red"\n"lime"\n"blue"\n')
                fname = fobj.name

            # Needs to be outside with so it's closed on windows
            self.reg.scan_dir(os.path.dirname(fname))
            name = os.path.splitext(os.path.basename(fobj.name))[0]

            assert name in self.reg
            assert self.reg[name] == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        finally:
            os.remove(fname)

    def test_read_file(self):
        'Test reading a colortable from a file'
        fobj = StringIO('(0., 0., 1.0)\n"red"\n"#0000FF" #Blue')

        self.reg.add_colortable(fobj, 'test_table')

        assert 'test_table' in self.reg
        assert self.reg['test_table'] == [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def test_get_colortable(self):
        'Test getting a colortable from the registry'
        true_colors = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
        self.reg['table'] = true_colors

        table = self.reg.get_colortable('table')
        assert table.N == 2
        assert table.colors == true_colors

    def test_get_steps(self):
        'Test getting a colortable and norm with appropriate steps'
        self.reg['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        norm, cmap = self.reg.get_with_steps('table', 5., 10.)
        assert cmap(norm(np.array([6.]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
        assert cmap(norm(np.array([14.9]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
        assert cmap(norm(np.array([15.1]))).tolist() == [[1.0, 0.0, 0.0, 1.0]]
        assert cmap(norm(np.array([26.]))).tolist() == [[0.0, 1.0, 0.0, 1.0]]

    def test_get_steps_negative_start(self):
        'Test for issue #81 (bad start for get with steps)'
        self.reg['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        norm, cmap = self.reg.get_with_steps('table', -10, 5)
        assert norm.vmin == -10
        assert norm.vmax == 5

    def test_get_boundaries(self):
        'Test getting a colortable with explicit boundaries'
        self.reg['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        norm, cmap = self.reg.get_with_boundaries('table', [0., 8., 10., 20.])
        assert cmap(norm(np.array([7.]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
        assert cmap(norm(np.array([9.]))).tolist() == [[1.0, 0.0, 0.0, 1.0]]
        assert cmap(norm(np.array([10.1]))).tolist() == [[0.0, 1.0, 0.0, 1.0]]


def test_gempak():
    'Test GEMPAK colortable conversion'
    infile = StringIO('''!   wvcolor.tbl
                         0      0      0
                       255    255    255
                       ''')
    outfile = StringIO()

    # Do the conversion
    convert_gempak_table(infile, outfile)

    # Reset and grab contents
    outfile.seek(0)
    result = outfile.read()

    assert result == '(0.000000, 0.000000, 0.000000)\n(1.000000, 1.000000, 1.000000)\n'
