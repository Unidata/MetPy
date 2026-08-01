from __future__ import annotations

from typing import Union

import pytest

from flexparser import flexparser as fp
from flexparser.testsuite.common import (
    CannotParseToFloat,
    Close,
    Comment,
    EqualFloat,
    MyBlock,
    MyRoot,
    NotAValidIdentifier,
    Open,
)


class MyBlock2(fp.Block[Open, Union[Comment, EqualFloat], Close, None]):
    pass


class MyRoot2(fp.RootBlock[Union[Comment, EqualFloat], None]):
    pass


FIRST_NUMBER = 1


def test_block_classes():
    assert tuple(MyBlock.opening_classes()) == (Open,)
    assert tuple(MyBlock.body_classes()) == (
        Comment,
        EqualFloat,
    )
    assert tuple(MyBlock.closing_classes()) == (Close,)

    assert tuple(MyRoot.opening_classes()) == ()
    assert tuple(MyRoot.body_classes()) == (
        Comment,
        EqualFloat,
    )
    assert tuple(MyRoot.closing_classes()) == ()

    assert tuple(MyBlock2.opening_classes()) == (Open,)
    assert tuple(MyBlock2.body_classes()) == (
        Comment,
        EqualFloat,
    )
    assert tuple(MyBlock2.closing_classes()) == (Close,)

    assert tuple(MyRoot2.opening_classes()) == ()
    assert tuple(MyRoot2.body_classes()) == (
        Comment,
        EqualFloat,
    )
    assert tuple(MyRoot2.closing_classes()) == ()


def test_formatting():
    obj = EqualFloat.from_string("a = 3.1")
    assert obj is not None
    assert obj.format_position == "N/A"
    obj.set_simple_position(10, 3, 7)
    assert obj.format_position == "10,3-10,10"
    assert (
        str(obj)
        == "EqualFloat(start_line=10, start_col=3, end_line=10, end_col=10, raw=None, a='a', b=3.1)"
    )

    obj = EqualFloat.from_string("%a = 3.1")
    assert obj is not None
    assert obj.format_position == "N/A"
    obj.set_simple_position(10, 3, 8)
    assert obj.format_position == "10,3-10,11"


def test_parse_equal_float():
    assert EqualFloat.from_string("a = 3.1") == EqualFloat("a", 3.1)
    assert EqualFloat.from_string("a") is None

    assert EqualFloat.from_string("%a = 3.1") == NotAValidIdentifier("%a")
    assert EqualFloat.from_string("a = 3f1") == CannotParseToFloat("3f1")

    obj = EqualFloat.from_string("a = 3f1")
    assert (
        str(obj)
        == "CannotParseToFloat(start_line=0, start_col=0, end_line=0, end_col=0, raw=None, value='3f1')"
    )


def test_consume_equal_float():
    f = lambda s: fp.StatementIterator(s, fp.SPLIT_EOL)
    assert EqualFloat.consume(f("a = 3.1"), None) == EqualFloat("a", 3.1).set_position(
        1, 0, 1, 7
    ).set_raw("a = 3.1")
    assert EqualFloat.consume(f("a"), None) is None

    assert EqualFloat.consume(f("%a = 3.1"), None) == NotAValidIdentifier(
        "%a"
    ).set_position(1, 0, 1, 8).set_raw("%a = 3.1")
    assert EqualFloat.consume(f("a = 3f1"), None) == CannotParseToFloat(
        "3f1"
    ).set_position(1, 0, 1, 7).set_raw("a = 3f1")


@pytest.mark.parametrize("klass", (MyRoot, MyRoot2))
def test_stream_block(klass):
    lines = "# hola\nx=1.0"
    si = fp.StatementIterator(lines, fp.SPLIT_EOL)

    mb = klass.consume_body_closing(fp.BOS(fp.Hash.nullhash()), si, None)
    assert isinstance(mb.opening, fp.BOS)
    assert isinstance(mb.closing, fp.EOS)
    body = tuple(mb.body)
    assert len(body) == 2
    assert body == (
        Comment("# hola")
        .set_position(FIRST_NUMBER + 0, 0, FIRST_NUMBER + 0, 6)
        .set_raw("# hola"),
        EqualFloat("x", 1.0)
        .set_position(FIRST_NUMBER + 1, 0, FIRST_NUMBER + 1, 5)
        .set_raw("x=1.0"),
    )
    assert tuple(mb) == (mb.opening, *body, mb.closing)
    assert not mb.has_errors


@pytest.mark.parametrize("klass", (MyRoot, MyRoot2))
def test_stream_block_error(klass):
    lines = "# hola\nx=1f0"
    si = fp.StatementIterator(lines, fp.SPLIT_EOL)

    mb = klass.consume_body_closing(fp.BOS(fp.Hash.nullhash()), si, None)
    assert isinstance(mb.opening, fp.BOS)
    assert isinstance(mb.closing, fp.EOS)
    body = tuple(mb.body)
    assert len(body) == 2
    assert body == (
        Comment("# hola").set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw("# hola"),
        CannotParseToFloat("1f0")
        .set_simple_position(FIRST_NUMBER + 1, 0, 5)
        .set_raw("x=1f0"),
    )
    assert tuple(mb) == (mb.opening, *body, mb.closing)
    assert mb.has_errors
    assert mb.errors == (
        CannotParseToFloat("1f0")
        .set_simple_position(FIRST_NUMBER + 1, 0, 5)
        .set_raw("x=1f0"),
    )


@pytest.mark.parametrize("klass", (MyBlock, MyBlock2))
def test_block(klass):
    lines = "@begin\n# hola\nx=1.0\n@end"
    si = fp.StatementIterator(lines, fp.SPLIT_EOL)

    mb = klass.consume(si, None)
    assert mb.opening == Open().set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw(
        "@begin"
    )
    assert mb.closing == Close().set_simple_position(FIRST_NUMBER + 3, 0, 4).set_raw(
        "@end"
    )
    body = tuple(mb.body)
    assert len(body) == 2
    assert mb.body == (
        Comment("# hola").set_simple_position(FIRST_NUMBER + 1, 0, 6).set_raw("# hola"),
        EqualFloat("x", 1.0)
        .set_simple_position(FIRST_NUMBER + 2, 0, 5)
        .set_raw("x=1.0"),
    )

    assert tuple(mb) == (mb.opening, *mb.body, mb.closing)
    assert not mb.has_errors


@pytest.mark.parametrize("klass", (MyBlock, MyBlock2))
def test_unfinished_block(klass):
    lines = "@begin\n# hola\nx=1.0"
    si = fp.StatementIterator(lines, fp.SPLIT_EOL)

    mb = klass.consume(si, None)
    assert mb.opening == Open().set_simple_position(FIRST_NUMBER + 0, 0, 6).set_raw(
        "@begin"
    )
    assert mb.closing == fp.UnexpectedEOS().set_simple_position(FIRST_NUMBER + 3, 0, 0)
    body = tuple(mb.body)
    assert len(body) == 2
    assert mb.body == (
        Comment("# hola").set_simple_position(FIRST_NUMBER + 1, 0, 6).set_raw("# hola"),
        EqualFloat("x", 1.0)
        .set_simple_position(FIRST_NUMBER + 2, 0, 5)
        .set_raw("x=1.0"),
    )

    assert tuple(mb) == (mb.opening, *mb.body, mb.closing)
    assert mb.has_errors


def test_not_proper_statement():
    class MySt(fp.ParsedStatement):
        pass

    with pytest.raises(NotImplementedError):
        MySt.from_string("a = 1")

    with pytest.raises(NotImplementedError):
        MySt.from_string_and_config("a = 1", None)
