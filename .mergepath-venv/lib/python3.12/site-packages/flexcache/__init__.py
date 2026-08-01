"""
    flexcache
    ~~~~~~~~~

    Classes for persistent caching and invalidating cached objects,
    which are built from a source object and a (potentially expensive)
    conversion function.

    :copyright: 2022 by flexcache Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from importlib.metadata import version

try:  # pragma: no cover
    __version__ = version("flexcache")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"

from .flexcache import (
    BaseHeader,
    BasicPythonHeader,
    DiskCache,
    DiskCacheByHash,
    DiskCacheByMTime,
    InvalidateByExist,
    InvalidateByMultiPathsMtime,
    InvalidateByPathMTime,
    NameByFields,
    NameByFileContent,
    NameByHashIter,
    NameByMultiPaths,
    NameByObj,
    NameByPath,
)

__all__ = (
    "__version__",
    "BaseHeader",
    "BasicPythonHeader",
    "NameByFields",
    "NameByFileContent",
    "NameByObj",
    "NameByPath",
    "NameByMultiPaths",
    "NameByHashIter",
    "DiskCache",
    "DiskCacheByHash",
    "DiskCacheByMTime",
    "InvalidateByExist",
    "InvalidateByPathMTime",
    "InvalidateByMultiPathsMtime",
)
