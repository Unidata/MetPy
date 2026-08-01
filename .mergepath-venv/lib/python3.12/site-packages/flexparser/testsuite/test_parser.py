from __future__ import annotations

import pytest

from flexparser import flexparser as fp
from flexparser.testsuite.common import (
    Comment,
    EqualFloat,
    MyBlock,
    MyParser,
    MyParserWithBlock,
)

FIRST_NUMBER = 1


@pytest.mark.parametrize(
    "content", (b"# hola\nx<>1.0", b"# hola\r\nx<>1.0", b"# hola\rx<>1.0")
)
def test_consume_err(content):
    myparser = MyParser(None)

    pf = myparser.parse_bytes(content).parsed_source
    assert isinstance(pf.opening, fp.BOS)
    assert isinstance(pf.closing, fp.EOS)
    body = tuple(pf.body)
    assert len(body) == 2
    assert body == (
        Comment("# hola").set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw("# hola"),
        fp.UnknownStatement()
        .set_simple_position(FIRST_NUMBER + 1, 0, 6)
        .set_raw("x<>1.0"),
    )
    assert tuple(pf) == (pf.opening, *body, pf.closing)
    assert pf.has_errors

    assert str(body[-1]) == "Could not parse 'x<>1.0' (2,0-2,6)"


@pytest.mark.parametrize(
    "content", (b"# hola\nx=1.0", b"# hola\r\nx=1.0", b"# hola\rx=1.0")
)
def test_consume(content):
    myparser = MyParser(None)

    pf = myparser.parse_bytes(content).parsed_source
    assert pf.start_line == 0
    assert pf.start_col == 0
    assert pf.end_line == 3
    assert pf.end_col == 0
    assert pf.format_position == "0,0-3,0"
    assert isinstance(pf.opening, fp.BOS)
    assert isinstance(pf.closing, fp.EOS)
    body = tuple(pf.body)
    assert len(body) == 2
    assert body == (
        Comment("# hola").set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw("# hola"),
        EqualFloat("x", 1.0)
        .set_simple_position(FIRST_NUMBER + 1, 0, 5)
        .set_raw("x=1.0"),
    )
    assert tuple(pf) == (pf.opening, *body, pf.closing)
    assert not pf.has_errors


@pytest.mark.parametrize("use_string", (True, False))
def test_parse(tmp_path, use_string):
    content = "# hola\nx=1.0"
    tmp_file = tmp_path / "bla.txt"
    tmp_file.write_text(content)
    myparser = MyParser(None)

    if use_string:
        psf = myparser.parse(str(tmp_file))
    else:
        psf = myparser.parse(tmp_file)

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


def test_unfinished_block(tmp_path):
    content = "@begin\n# hola\nx=1.0"

    tmp_file = tmp_path / "bla.txt"
    tmp_file.write_text(content)
    myparser = MyParserWithBlock(None)

    psf = myparser.parse(tmp_file)
    assert psf.has_errors
    assert psf.config is None
    assert psf.parsed_source.opening.mtime == tmp_file.stat().st_mtime
    assert psf.parsed_source.opening.path == tmp_file
    assert tuple(psf.errors()) == (
        fp.UnexpectedEOS().set_simple_position(FIRST_NUMBER + 3, 0, 0),
    )
    assert psf.location == tmp_file

    mb = psf.parsed_source
    assert isinstance(mb.opening, fp.BOS)
    assert isinstance(mb.closing, fp.EOS)
    body = tuple(mb.body)
    assert len(body) == 1
    assert isinstance(body[0], MyBlock)
    assert body[0].closing == fp.UnexpectedEOS().set_simple_position(
        FIRST_NUMBER + 3, 0, 0
    )
    assert body[0].body == (
        Comment("# hola").set_simple_position(FIRST_NUMBER + 1, 0, 6).set_raw("# hola"),
        EqualFloat("x", 1.0)
        .set_simple_position(FIRST_NUMBER + 2, 0, 5)
        .set_raw("x=1.0"),
    )


@pytest.mark.parametrize("use_resource", (True, False))
def test_parse_resource(use_resource):
    myparser = MyParser(None)

    location = ("flexparser.testsuite", "bla2.txt")

    if use_resource:
        psf = myparser.parse_resource(*location)
    else:
        psf = myparser.parse(location)

    assert not psf.has_errors
    assert psf.config is None
    assert tuple(psf.errors()) == ()

    if use_resource:
        assert psf.location == location

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
