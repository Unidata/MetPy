import glob
import os.path
import posixpath
import matplotlib.colors as mcolors
from pkg_resources import resource_listdir, resource_stream

TABLE_EXT = '.tbl'


def read_colortable(fobj):
    r'''Read colortable information from a file

    Reads a colortable, which consists of one color per line of the file, where
    a color can be one of: a tuple of 3 floats, a string with a HTML color name,
    or a string with a HTML hex color.

    Parameters
    ----------
    fobj : a file-like object
        A file-like object to read the colors from

    Returns
    -------
    List of tuples
        A list of the RGB color values, where each RGB color is a tuple of 3 floats in the
        range of [0, 1].
    '''
    return [mcolors.colorConverter.to_rgb(eval(line)) for line in fobj]


class ColortableRegistry(dict):
    r'''Manages the collection of colortables.

    Provides access to colortables, read collections of files, and generates
    matplotlib's Normalize instances to go with the colortable.
    '''

    def scan_resource(self, pkg, path):
        r'''Scan a resource directory for colortable files and add them to the registry

        Parameters
        ----------
        pkg : str
            The package containing the resource directory
        path : str
            The path to the directory with the colortables
        '''

        for fname in resource_listdir(pkg, path):
            if fname.endswith(TABLE_EXT):
                self.add_colortable(resource_stream(pkg, posixpath.join(path, fname)),
                                    posixpath.splitext(posixpath.basename(fname))[0])

    def scan_dir(self, path):
        r'''Scan a directory on disk for colortable files and add them to the registry

        Parameters
        ----------
        path : str
            The path to the directory with the colortables
        '''

        for fname in glob.glob(os.path.join(path, '*' + TABLE_EXT)):
            if os.path.isfile(fname):
                with open(fname, 'r') as fobj:
                    self.add_colortable(fobj, os.path.splitext(os.path.basename(fname))[0])

    def add_colortable(self, fobj, name):
        r'''Add a colortable from a file to the registry

        Parameters
        ----------
        fobj : file-like object
            The file to read the colortable from
        name : str
            The name under which the colortable will be stored
        '''

        self[name] = read_colortable(fobj)

    def get_with_limits(self, name, start, step):
        r'''Get a colortable from the registry with a corresponding norm.

        Builds a `matplotlib.colors.BoundaryNorm` using `start`, `step`, and
        the number of colors, based on the colortable obtained from `name`.

        Parameters
        ----------
        name : str
            The name under which the colortable will be stored
        start : float
            The starting boundary
        step : float
            The step between boundaries

        Returns
        -------
        `matplotlib.colors.BoundaryNorm`, `matplotlib.colors.ListedColormap`
            The boundary norm based on `start` and `step` with the number of colors
            from the number of entries matching the colortable, and the colortable itself.
        '''

        import numpy as np
        cmap = mcolors.ListedColormap(self[name])
        boundaries = np.linspace(start, step * cmap.N, cmap.N)
        return mcolors.BoundaryNorm(boundaries, cmap.N), cmap

    def get_with_boundaries(self, name, boundaries):
        r'''Get a colortable from the registry with a corresponding norm.

        Builds a `matplotlib.colors.BoundaryNorm` using `boundaries`.

        Parameters
        ----------
        name : str
            The name under which the colortable will be stored
        boundaries : array_like
            The list of boundaries for the norm

        Returns
        -------
        `matplotlib.colors.BoundaryNorm`, `matplotlib.colors.ListedColormap`
            The boundary norm based on `boundaries`, and the colortable itself.
        '''

        cmap = mcolors.ListedColormap(self[name])
        return mcolors.BoundaryNorm(boundaries, cmap.N), cmap

    def get_colortable(self, name):
        r'''Get a colortable from the registry

        Parameters
        ----------
        name : str
            The name under which the colortable will be stored

        Returns
        -------
        `matplotlib.colors.ListedColormap`
            The colortable corresponding to `name`
        '''

        return mcolors.ListedColormap(dict.get(self, name))


registry = ColortableRegistry()
registry.scan_resource('metpy.plots', 'colortables')
registry.scan_dir(os.path.curdir)
