# Copyright (c) 2008,2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Collection of generally useful utility code from the cookbook."""

import os
import os.path
import shutil
import sys
from urllib.error import URLError
import urllib.request

from matplotlib.cbook import iterable

from . import __version__


try:
    string_type = basestring
except NameError:
    string_type = str


# TODO: This can go away when we remove Python 2
def is_string_like(s):
    """Check if an object is a string."""
    return isinstance(s, string_type)


def get_test_data(fname, as_file_obj=True):
    """Access a file from MetPy's collection of test data."""
    cache = CacheData()
    path = cache.get_data(fname)

    # If we want a file object, open it, trying to guess whether this should be binary mode
    # or not
    if as_file_obj:
        return open(path, 'rb')

    return path


class CacheData(object):
    """Maintain a cache of test/example data as well as other downloaded resources.

    Implements cache versioning, custom locations, and is MetPy version aware.
    """

    cache_version = 1

    def __init__(self, path=None):
        """Initialize :class:`CacheData`."""
        # Determine where the cache lives.
        self.cache_home = self.get_cache_location(path)
        self.cache_version_file_path = os.path.join(self.cache_home, 'cache_version.txt')

        # If there's not an existing cache, make one.
        if not os.path.exists(self.cache_home):
            self.initialize_cache()

        # Check if the cache is outdated. If so, clear and reinitialize.
        if self.cache_outdated():
            self.clear_cache()
            self.initialize_cache()

    def cache_outdated(self):
        """Check to see if the cache version is missing or out of date.

        If somehow the cache version file is missing or does not match the `cache_version`
        property, the cache is to be considered outdated.

        Returns
        -------
        bool: If cache is outdated.

        """
        # If the cache file is gone or not equal to the cache version locally, it's outdated.
        if not os.path.exists(self.cache_version_file_path):
            return True

        with open(self.cache_version_file_path, 'r') as f:
            current_cache_version = int(f.readline())
        if current_cache_version != self.cache_version:
            return True
        else:
            return False

    def initialize_cache(self):
        """Create a new cache.

        Creates a new cache folder at the location cache_home.
        """
        os.makedirs(self.cache_home)
        os.makedirs(os.path.join(self.cache_home, 'nids'))
        with open(self.cache_version_file_path, 'w') as f:
            f.write('{}\n'.format(self.cache_version))

    def get_cache_location(self, path=None):
        """Get the cache location.

        Determine where the cache should be located. User specified path is top priority,
        followed by a location defined by the environment variable METPY_CACHE_LOCATION.
        Finally, if there is a staticdata directory (because this is a git checkout) well
        use that. Otherwise, use the operating system default location.

        Parameters
        ----------
        path: str
            User defined location for the MetPy data cache.

        Returns
        -------
        cache_path: Cache location

        """
        # If a user provides a path, it takes priority
        if path:
            return path

        # Look for the staticdata directory (i.e. this is a git checkout)
        if os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'staticdata')):
            cache_path = os.path.join(os.path.dirname(__file__), '..', 'staticdata')

        # If there is no static data directory, use the OS default
        else:
            # Get the cache location for this OS
            cache_path = self.get_os_cache_path()

        # Look for a user-defined cache location in the environment variables.
        if os.environ.get('METPY_CACHE_LOCATION'):
            cache_path = os.environ.get('METPY_CACHE_LOCATION')

        # Make sure we expand ~ to the user home
        cache_path = os.path.expanduser(cache_path)

        return cache_path

    @staticmethod
    def get_os_cache_path():
        """Determine cache path based on operating system."""
        platform = sys.platform

        if platform == 'darwin':
            cache_path = '~/Library/Caches/MetPy'
        elif platform == 'win32':
            cache_path = '~/AppData/Local/MetPy/cache'
        else:  # *NIX
            cache_path = '~/.cache/MetPy'

        return os.path.expanduser(cache_path)

    def get_data(self, fname):
        """Get data from the cache."""
        file_path = os.path.join(self.cache_home, fname)

        # If we don't have the file locally, try to download it
        if not os.path.exists(file_path):
            self.download_data(fname)

        return file_path

    def download_data(self, fname):
        """Download remote data to the cache."""
        github_url = self.github_data_url(fname)

        cache_fname = os.path.join(self.cache_home, fname)
        cache_subpath = os.path.split(cache_fname)[0]

        # Make sure if it is a subdirectory that it exists before pulling down data
        if not os.path.exists(cache_subpath):
            os.makedirs(cache_subpath)

        try:
            urllib.request.urlretrieve(github_url, cache_fname)
        except URLError:
            raise URLError('Unable to retrieve {}'.format(github_url))

    @staticmethod
    def github_data_url(fname):
        """Make the GitHub URL for this version of MetPy's cached file."""
        if '+' in __version__:
            gh_tag = 'master'
        else:
            gh_tag = 'v{}'.format(__version__)
        github_data_path = 'https://github.com/Unidata/MetPy/raw/{}/staticdata'.format(gh_tag)
        return github_data_path + '/' + fname

    def clear_cache(self):
        """Remove all cached data for MetPy."""
        shutil.rmtree(self.cache_home)


__all__ = ('get_test_data', 'is_string_like', 'iterable', 'CacheData')
