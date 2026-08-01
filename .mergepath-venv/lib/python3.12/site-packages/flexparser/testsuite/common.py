from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias  # noqa
else:
    from typing_extensions import TypeAlias  # noqa


if sys.version_info >= (3, 11):
    from typing import Self  # noqa
else:
    from typing_extensions import Self  # noqa

from flexparser import flexparser as fp


class NotAValidIdentifier(fp.ParsingError):
    value: str

    def __init__(self, value: str) -> None:
        self.value = value

    def custom_values_str(self) -> str:
        return f"value='{self.value}'"


class CannotParseToFloat(fp.ParsingError):
    value: str

    def __init__(self, value: str) -> None:
        self.value = value

    def custom_values_str(self) -> str:
        return f"value='{self.value}'"


@dataclass(frozen=True)
class Open(fp.ParsedStatement[None]):
    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[Self]:
        if s == "@begin":
            return cls()
        return None


@dataclass(frozen=True)
class Close(fp.ParsedStatement[None]):
    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[Self]:
        if s == "@end":
            return cls()
        return None


@dataclass(frozen=True)
class Comment(fp.ParsedStatement[None]):
    s: str

    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[Self]:
        if s.startswith("#"):
            return cls(s)
        return None


@dataclass(frozen=True)
class EqualFloat(fp.ParsedStatement[None]):
    a: str
    b: float

    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[Self]:
        if "=" not in s:
            return None

        a, b = s.split("=")
        a = a.strip()
        b = b.strip()

        if not str.isidentifier(a):
            return NotAValidIdentifier(a)

        try:
            b = float(b)
        except Exception:
            return CannotParseToFloat(b)

        return cls(a, b)


class MyBlock(fp.Block[Open, Union[Comment, EqualFloat], Close, None]):
    pass


class MyRoot(fp.RootBlock[Union[Comment, EqualFloat], None]):
    pass


class MyParser(fp.Parser[MyRoot, None]):
    pass


class MyRootWithBlock(fp.RootBlock[Union[Comment, EqualFloat, MyBlock], None]):
    pass


class MyParserWithBlock(fp.Parser[MyRootWithBlock, None]):
    pass
