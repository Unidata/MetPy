import os.path
import tempfile
try:
    from StringIO import StringIO
    buffer_args = dict(bufsize=0)
except ImportError:
    from io import StringIO
    buffer_args = dict(buffering=1)

from nose.tools import eq_
from metpy.plots.ctables import ColortableRegistry


class TestColortableRegistry(object):
    def test_package_resource(self):
        reg = ColortableRegistry()
        reg.scan_resource('metpy.plots', 'nexrad_tables')
        assert 'cc_table' in reg

    def test_scan_dir(self):
        with tempfile.NamedTemporaryFile(mode='w', dir='.', suffix='.tbl',
                                         **buffer_args) as fobj:
            fobj.write('"red"\n"lime"\n"blue"\n')
            reg = ColortableRegistry()
            reg.scan_dir(os.path.dirname(fobj.name))
        name = os.path.splitext(os.path.basename(fobj.name))[0]

        assert name in reg
        eq_(reg[name], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])

    def test_read_file(self):
        fobj = StringIO('(0., 0., 1.0)\n"red"\n"#0000FF"')

        reg = ColortableRegistry()
        reg.add_colortable(fobj, 'test_table')

        assert 'test_table' in reg
        assert reg['test_table'] == [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
