"""
flexparser
~~~~~~~~~

Classes and functions to create parsers.

The idea is quite simple. You write a class for every type of content
(called here ``ParsedStatement``) you need to parse. Each class should
have a ``from_string`` constructor. We used extensively the ``typing``
module to make the output structure easy to use and less error prone.

:copyright: 2022 by flexparser Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from importlib.metadata import version

try:  # pragma: no cover
    __version__ = version("flexparser")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"


from .flexparser import (
    BOF,
    BOR,
    BOS,
    EOS,
    SPLIT_EOL,
    Block,
    DelimiterAction,
    DelimiterInclude,
    IncludeStatement,
    NullableParsedResult,
    ParsedProject,
    ParsedSource,
    ParsedStatement,
    Parser,
    ParsingError,
    RootBlock,
    StatementIterator,
    UnexpectedEOS,
    UnknownStatement,
    parse,
    parse_bytes,
)

# Deprecate in 0.3
UnexpectedEOF = UnexpectedEOS

__all__ = (
    "__version__",
    "Block",
    "DelimiterAction",
    "DelimiterInclude",
    "IncludeStatement",
    "ParsedStatement",
    "Parser",
    "ParsingError",
    "RootBlock",
    "StatementIterator",
    "UnexpectedEOF",
    "UnexpectedEOS",
    "UnknownStatement",
    "parse",
    "ParsedSource",
    "ParsedProject",
    "SPLIT_EOL",
    "BOF",
    "BOR",
    "BOS",
    "EOS",
    "parse_bytes",
    "NullableParsedResult",
)
