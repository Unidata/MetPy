# Copyright (c) 2015,2016,2017,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `ctables` module."""

import os.path
import tempfile
try:
    buffer_args = {'bufsize': 0}
    from StringIO import StringIO
except ImportError:
    buffer_args = {'buffering': 1}
    from io import StringIO

import numpy as np
import pytest

from metpy.deprecation import MetpyDeprecationWarning
from metpy.plots.ctables import ColortableRegistry, convert_gempak_table


@pytest.fixture()
def registry():
    """Set up a registry for use by the tests."""
    return ColortableRegistry()


def test_package_resource(registry):
    """Test registry scanning package resource."""
    registry.scan_resource('metpy.plots', 'nexrad_tables')
    assert 'cc_table' in registry


def test_scan_dir(registry):
    """Test registry scanning a directory and ignoring files it can't handle ."""
    try:
        kwargs = {'mode': 'w', 'dir': '.', 'suffix': '.tbl', 'delete': False}
        kwargs.update(**buffer_args)
        with tempfile.NamedTemporaryFile(**kwargs) as fobj:
            fobj.write('"red"\n"lime"\n"blue"\n')
            fname = fobj.name

        # Unrelated table file that *should not* impact the scan
        with tempfile.NamedTemporaryFile(**kwargs) as fobj:
            fobj.write('PADK     704540 ADAK NAS\n')
            bad_file = fobj.name

        # Needs to be outside with so it's closed on windows
        registry.scan_dir(os.path.dirname(fname))
        name = os.path.splitext(os.path.basename(fname))[0]

        assert name in registry
        assert registry[name] == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    finally:
        os.remove(fname)
        os.remove(bad_file)


def test_read_file(registry):
    """Test reading a colortable from a file."""
    fobj = StringIO('(0., 0., 1.0)\n"red"\n"#0000FF" #Blue')

    registry.add_colortable(fobj, 'test_table')

    assert 'test_table' in registry
    assert registry['test_table'] == [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def test_read_bad_file(registry):
    """Test what error results when reading a malformed file."""
    with pytest.raises(RuntimeError):
        fobj = StringIO('PADK     704540 ADAK NAS                         '
                        'AK US  5188 -17665     4  0')
        registry.add_colortable(fobj, 'sfstns')


def test_get_colortable(registry):
    """Test getting a colortable from the registry."""
    true_colors = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
    registry['table'] = true_colors

    table = registry.get_colortable('table')
    assert table.N == 2
    assert table.colors == true_colors


def test_get_steps(registry):
    """Test getting a colortable and norm with appropriate steps."""
    registry['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    norm, cmap = registry.get_with_steps('table', 5., 10.)
    assert cmap(norm(np.array([6.]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert cmap(norm(np.array([14.9]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert cmap(norm(np.array([15.1]))).tolist() == [[1.0, 0.0, 0.0, 1.0]]
    assert cmap(norm(np.array([26.]))).tolist() == [[0.0, 1.0, 0.0, 1.0]]


def test_get_steps_negative_start(registry):
    """Test bad start for get with steps (issue #81)."""
    registry['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    norm, _ = registry.get_with_steps('table', -10, 5)
    assert norm.vmin == -10
    assert norm.vmax == 5


def test_get_range(registry):
    """Test getting a colortable and norm with appropriate range."""
    registry['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    norm, cmap = registry.get_with_range('table', 5., 35.)
    assert cmap(norm(np.array([6.]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert cmap(norm(np.array([14.9]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert cmap(norm(np.array([15.1]))).tolist() == [[1.0, 0.0, 0.0, 1.0]]
    assert cmap(norm(np.array([26.]))).tolist() == [[0.0, 1.0, 0.0, 1.0]]


def test_get_boundaries(registry):
    """Test getting a colortable with explicit boundaries."""
    registry['table'] = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    norm, cmap = registry.get_with_boundaries('table', [0., 8., 10., 20.])
    assert cmap(norm(np.array([7.]))).tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert cmap(norm(np.array([9.]))).tolist() == [[1.0, 0.0, 0.0, 1.0]]
    assert cmap(norm(np.array([10.1]))).tolist() == [[0.0, 1.0, 0.0, 1.0]]


def test_gempak():
    """Test GEMPAK colortable conversion."""
    infile = StringIO("""!   wvcolor.tbl
                         0      0      0
                       255    255    255
                       """)
    outfile = StringIO()

    # Do the conversion
    convert_gempak_table(infile, outfile)

    # Reset and grab contents
    outfile.seek(0)
    result = outfile.read()

    assert result == '(0.000000, 0.000000, 0.000000)\n(1.000000, 1.000000, 1.000000)\n'


def test_viridis(registry):
    """Test viridis deprecation warning."""
    with pytest.warns(MetpyDeprecationWarning):
        registry.scan_resource('metpy.plots', 'colortable_files')
        registry.get_colortable('viridis')
