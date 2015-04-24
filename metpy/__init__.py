# What do we want to pull into the top-level namespace?

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
