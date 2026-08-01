from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass

import pytest

from flexparser import flexparser as fp
from flexparser.testsuite.common import (
    Close,
    Comment,
    EqualFloat,
    MyBlock,
    MyParser,
    MyRoot,
    Open,
)

FIRST_NUMBER = 1


def _bosser(content: bytes):
    return fp.BOS(fp.Hash.from_bytes(hashlib.blake2b, content))


def _compare(arr1, arr2):
    assert len(arr1) == len(arr2)
    for a1, a2 in zip(arr1, arr2):
        if isinstance(a1, fp.BOS) and isinstance(a2, fp.BOS):
            assert a1.content_hash == a2.content_hash
        else:
            assert a1 == a2, str(a1) + " == " + str(a2)


def test_locator():
    this_file = pathlib.Path(__file__)

    with pytest.raises(ValueError):
        # Cannot use absolute path as target.
        assert fp.default_locator(this_file, "/temp/bla.txt")

    with pytest.raises(TypeError):
        assert fp.default_locator(str(this_file), "bla.txt")

    with pytest.raises(TypeError):
        assert fp.default_locator(str(this_file), "/temp/bla.txt")

    assert fp.default_locator(this_file, "bla.txt") == this_file.parent / "bla.txt"
    assert (
        fp.default_locator(this_file.parent, "bla.txt") == this_file.parent / "bla.txt"
    )

    with pytest.raises(ValueError):
        assert (
            fp.default_locator(this_file.parent, "../bla.txt")
            == this_file.parent / "bla.txt"
        )

    assert fp.default_locator(("pack", "nam"), "bla") == ("pack", "bla")


@pytest.mark.parametrize("definition", [MyRoot, (Comment, EqualFloat), MyParser])
def test_parse1(tmp_path, definition):
    content = b"# hola\nx=1.0"
    tmp_file = tmp_path / "bla.txt"
    tmp_file.write_bytes(content)

    pp = fp.parse(tmp_file, definition)

    assert len(pp) == 1

    psf = pp[list(pp.keys())[0]]
    assert not psf.has_errors
    assert psf.config is None
    assert psf.parsed_source.opening.mtime == tmp_file.stat().st_mtime
    assert psf.parsed_source.opening.path == tmp_file
    assert tuple(psf.errors()) == ()
    assert psf.location == tmp_file

    # TODO:
    # assert psf.content_hash == hashlib.sha1(content.encode("utf-8")).hexdigest()

    mb = psf.parsed_source
    assert isinstance(mb.opening, fp.BOS)
    assert isinstance(mb.closing, fp.EOS)
    body = tuple(mb.body)
    assert len(body) == 2
    assert body == (
        Comment("# hola").set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw("# hola"),
        EqualFloat("x", 1.0)
        .set_simple_position(FIRST_NUMBER + 1, 0, 5)
        .set_raw("x=1.0"),
    )
    assert tuple(mb) == (mb.opening, *body, mb.closing)
    assert not mb.has_errors

    _compare(
        tuple(pp.iter_statements()),
        (
            _bosser(content).set_simple_position(FIRST_NUMBER + 0, 0, 0),
            Comment("# hola")
            .set_simple_position(FIRST_NUMBER + 0, 0, 6)
            .set_raw("# hola"),
            EqualFloat("x", 1.0)
            .set_simple_position(FIRST_NUMBER + 1, 0, 5)
            .set_raw("x=1.0"),
            fp.EOS().set_simple_position(FIRST_NUMBER + 2, 0, 0),
        ),
    )


@pytest.mark.parametrize("definition", [MyRoot, EqualFloat, MyParser])
def test_parse2(tmp_path, definition):
    content = b"y = 2.0\nx=1.0"
    tmp_file = tmp_path / "bla.txt"
    tmp_file.write_bytes(content)

    pp = fp.parse(tmp_file, definition)

    assert len(pp) == 1

    assert None in pp
    psf = pp[None]
    assert not psf.has_errors
    assert psf.config is None
    assert psf.parsed_source.opening.mtime == tmp_file.stat().st_mtime
    assert psf.parsed_source.opening.path == tmp_file
    assert tuple(psf.errors()) == ()
    assert psf.location == tmp_file

    # TODO:
    # assert psf.content_hash == hashlib.sha1(content.encode("utf-8")).hexdigest()

    mb = psf.parsed_source
    assert isinstance(mb.opening, fp.BOS)
    assert isinstance(mb.closing, fp.EOS)
    body = tuple(mb.body)
    assert len(body) == 2
    assert body == (
        EqualFloat("y", 2.0)
        .set_simple_position(FIRST_NUMBER + 0, 0, 7)
        .set_raw("y = 2.0"),
        EqualFloat("x", 1.0)
        .set_simple_position(FIRST_NUMBER + 1, 0, 5)
        .set_raw("x=1.0"),
    )
    assert tuple(mb) == (mb.opening, *body, mb.closing)
    assert not mb.has_errors

    _compare(
        tuple(pp.iter_statements()),
        (
            _bosser(content).set_simple_position(FIRST_NUMBER + 0, 0, 0),
            EqualFloat("y", 2.0)
            .set_simple_position(FIRST_NUMBER + 0, 0, 7)
            .set_raw("y = 2.0"),
            EqualFloat("x", 1.0)
            .set_simple_position(FIRST_NUMBER + 1, 0, 5)
            .set_raw("x=1.0"),
            fp.EOS().set_simple_position(FIRST_NUMBER + 2, 0, 0),
        ),
    )


@pytest.mark.parametrize(
    "definition",
    [
        MyBlock,
    ],
)
def test_parse3(tmp_path, definition):
    content = b"@begin\ny = 2.0\nx=1.0\n@end"
    tmp_file = tmp_path / "bla.txt"
    tmp_file.write_bytes(content)

    pp = fp.parse(tmp_file, definition)
    assert not pp.has_errors
    assert len(pp) == 1
    assert tuple(pp.errors()) == ()

    assert None in pp
    psf = pp[None]
    assert not psf.has_errors
    assert psf.config is None
    assert psf.parsed_source.opening.mtime == tmp_file.stat().st_mtime
    assert psf.parsed_source.opening.path == tmp_file
    assert tuple(psf.errors()) == ()
    assert psf.location == tmp_file

    # TODO:
    # assert psf.content_hash == hashlib.sha1(content.encode("utf-8")).hexdigest()

    mb = psf.parsed_source
    assert isinstance(mb.opening, fp.BOS)
    assert isinstance(mb.closing, fp.EOS)
    body = tuple(mb.body)
    assert len(body) == 1
    mb = body[0]

    assert mb.start_line == 1
    assert mb.start_col == 0
    assert mb.end_line == 4
    assert mb.end_col == 4
    assert mb.format_position == "1,0-4,4"

    assert mb.opening == Open().set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw(
        "@begin"
    )
    assert tuple(mb.body) == (
        EqualFloat("y", 2.0)
        .set_simple_position(FIRST_NUMBER + 1, 0, 7)
        .set_raw("y = 2.0"),
        EqualFloat("x", 1.0)
        .set_simple_position(FIRST_NUMBER + 2, 0, 5)
        .set_raw("x=1.0"),
    )
    assert mb.closing == Close().set_simple_position(FIRST_NUMBER + 3, 0, 4).set_raw(
        "@end"
    )
    assert not mb.has_errors

    _compare(
        tuple(pp.iter_statements()),
        (
            _bosser(content).set_simple_position(FIRST_NUMBER + 0, 0, 0),
            Open().set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw("@begin"),
            EqualFloat("y", 2.0)
            .set_simple_position(FIRST_NUMBER + 1, 0, 7)
            .set_raw("y = 2.0"),
            EqualFloat("x", 1.0)
            .set_simple_position(FIRST_NUMBER + 2, 0, 5)
            .set_raw("x=1.0"),
            Close().set_simple_position(FIRST_NUMBER + 3, 0, 4).set_raw("@end"),
            fp.EOS().set_simple_position(FIRST_NUMBER + 4, 0, 0),
        ),
    )


def test_include_file(tmp_path):
    @dataclass(frozen=True)
    class Include(fp.IncludeStatement[None]):
        value: str

        @property
        def target(self) -> str:
            return "bla2.txt"

        @classmethod
        def from_string(cls, s: str):
            if s.startswith("include"):
                return cls(s[len("include ") :].strip())

    content1 = b"include bla2.txt\n# chau"
    content2 = b"# hola\nx=1.0"

    tmp_file1 = tmp_path / "bla1.txt"
    tmp_file2 = tmp_path / "bla2.txt"
    tmp_file1.write_bytes(content1)
    tmp_file2.write_bytes(content2)

    pp = fp.parse(tmp_file1, (Include, Comment, EqualFloat), None)

    assert None in pp
    assert (tmp_file1, "bla2.txt") in pp

    assert len(pp) == 2

    _compare(
        tuple(pp.iter_statements()),
        (
            _bosser(content1).set_simple_position(FIRST_NUMBER + 0, 0, 0),
            # Include
            _bosser(content2).set_simple_position(FIRST_NUMBER + 0, 0, 6),
            Comment("# hola")
            .set_simple_position(FIRST_NUMBER + 0, 0, 6)
            .set_raw("# hola"),
            EqualFloat("x", 1.0)
            .set_simple_position(FIRST_NUMBER + 1, 0, 5)
            .set_raw("x=1.0"),
            fp.EOS().set_simple_position(FIRST_NUMBER + 2, 0, 0),
            Comment("# chau")
            .set_simple_position(FIRST_NUMBER + 1, 0, 6)
            .set_raw("# chau"),
            fp.EOS().set_simple_position(FIRST_NUMBER + 2, 0, 0),
        ),
    )


def test_resources(tmp_path):
    @dataclass(frozen=True)
    class Include(fp.IncludeStatement[None]):
        value: str

        @property
        def target(self) -> str:
            return "bla2.txt"

        @classmethod
        def from_string(cls, s: str):
            if s.startswith("include"):
                return cls(s[len("include ") :].strip())

    # see files included in the textsuite.
    content1 = b"include bla2.txt\n# chau\n"
    content2 = b"# hola\nx=1.0\n"

    pp = fp.parse(("flexparser.testsuite", "bla1.txt"), (Include, Comment, EqualFloat))

    assert len(pp) == 2

    _compare(
        tuple(pp.iter_statements()),
        (
            _bosser(content1).set_simple_position(FIRST_NUMBER + 0, 0, 0),
            # include
            _bosser(content2).set_simple_position(FIRST_NUMBER + 0, 0, 0),
            Comment("# hola")
            .set_simple_position(FIRST_NUMBER + 0, 0, 6)
            .set_raw("# hola"),
            EqualFloat("x", 1.0)
            .set_simple_position(FIRST_NUMBER + 1, 0, 5)
            .set_raw("x=1.0"),
            fp.EOS().set_simple_position(FIRST_NUMBER + 2, 0, 0),
            Comment("# chau")
            .set_simple_position(FIRST_NUMBER + 1, 0, 6)
            .set_raw("# chau"),
            fp.EOS().set_simple_position(FIRST_NUMBER + 2, 0, 0),
        ),
    )
