import os.path
import tempfile
try:
    from StringIO import StringIO
    buffer_args = dict(bufsize=0)
except ImportError:
    from io import StringIO
    buffer_args = dict(buffering=1)

import numpy as np
from nose.tools import eq_
from metpy.plots.ctables import ColortableRegistry


class TestColortableRegistry(object):
    def setUp(self):  # noqa
        self.reg = ColortableRegistry()

    def test_package_resource(self):
        self.reg.scan_resource('metpy.plots', 'nexrad_tables')
        assert 'cc_table' in self.reg

    def test_scan_dir(self):
        with tempfile.NamedTemporaryFile(mode='w', dir='.', suffix='.tbl',
                                         **buffer_args) as fobj:
            fobj.write('"red"\n"lime"\n"blue"\n')
            self.reg.scan_dir(os.path.dirname(fobj.name))
        name = os.path.splitext(os.path.basename(fobj.name))[0]

        assert name in self.reg
        eq_(self.reg[name], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])

    def test_read_file(self):
        fobj = StringIO('(0., 0., 1.0)\n"red"\n"#0000FF" #Blue')

        self.reg.add_colortable(fobj, 'test_table')

        assert 'test_table' in self.reg
        assert self.reg['test_table'] == [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    def test_get_colortable(self):
        true_colors = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
        self.reg['table'] = true_colors

        table = self.reg.get_colortable('table')
        eq_(table.N, 2)
        eq_(table.colors, true_colors)

    def test_get_steps(self):
        self.reg['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        norm, cmap = self.reg.get_with_steps('table', 5., 10.)
        eq_(cmap(norm(np.array([6.]))).tolist(), [[0.0, 0.0, 1.0, 1.0]])
        eq_(cmap(norm(np.array([14.9]))).tolist(), [[0.0, 0.0, 1.0, 1.0]])
        eq_(cmap(norm(np.array([15.1]))).tolist(), [[1.0, 0.0, 0.0, 1.0]])
        eq_(cmap(norm(np.array([26.]))).tolist(), [[0.0, 1.0, 0.0, 1.0]])

    def test_get_boundaries(self):
        self.reg['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        norm, cmap = self.reg.get_with_boundaries('table', [0., 8., 10., 20.])
        eq_(cmap(norm(np.array([7.]))).tolist(), [[0.0, 0.0, 1.0, 1.0]])
        eq_(cmap(norm(np.array([9.]))).tolist(), [[1.0, 0.0, 0.0, 1.0]])
        eq_(cmap(norm(np.array([10.1]))).tolist(), [[0.0, 1.0, 0.0, 1.0]])
