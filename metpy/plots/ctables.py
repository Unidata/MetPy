import glob
import os.path
import posixpath
import matplotlib.colors as mcolors
from pkg_resources import resource_listdir, resource_stream

TABLE_EXT = '.tbl'


def read_colortable(fobj):
    return [mcolors.colorConverter.to_rgb(eval(line)) for line in fobj]


class ColortableRegistry(dict):
    def scan_resource(self, pkg, path):
        for fname in resource_listdir(pkg, path):
            if fname.endswith(TABLE_EXT):
                self.add_colortable(resource_stream(pkg, posixpath.join(path, fname)),
                                    posixpath.splitext(posixpath.basename(fname))[0])

    def scan_dir(self, path):
        for fname in glob.glob(os.path.join(path, '*' + TABLE_EXT)):
            if os.path.isfile(fname):
                with open(fname, 'r') as fobj:
                    self.add_colortable(fobj, os.path.splitext(os.path.basename(fname))[0])

    def add_colortable(self, fobj, name):
        self[name] = read_colortable(fobj)

    def get_with_limits(self, name, start, step):
        import numpy as np
        cmap = mcolors.ListedColormap(self[name])
        boundaries = np.linspace(start, step * cmap.N, cmap.N)
        return mcolors.BoundaryNorm(boundaries, cmap.N), cmap

    def get_with_boundaries(self, name, boundaries):
        cmap = mcolors.ListedColormap(self[name])
        return mcolors.BoundaryNorm(boundaries, cmap.N), cmap

    def get_colortable(self, name):
        return mcolors.ListedColormap(dict.get(self, name))


registry = ColortableRegistry()
registry.scan_resource('metpy.plots', 'colortables')
registry.scan_dir(os.path.curdir)
