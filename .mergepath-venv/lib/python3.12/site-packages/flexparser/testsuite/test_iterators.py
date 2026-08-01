from __future__ import annotations

import pytest

import flexparser.flexparser as fp


@pytest.mark.parametrize(
    "delimiters,content,expected",
    [
        # ### 0
        (
            {},
            "Testing # 123",
            ((0, 0, 0, 13, "Testing # 123"),),
        ),
        # ### 1
        (
            {"#": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.STOP_PARSING)},
            "Testing # 123",
            ((0, 0, 0, 8, "Testing "),),
        ),
        # ### 2
        (
            {"#": (fp.DelimiterInclude.SPLIT_AFTER, fp.DelimiterAction.STOP_PARSING)},
            "Testing # 123",
            ((0, 0, 0, 9, "Testing #"),),
        ),
        # ### 3
        (
            {"#": (fp.DelimiterInclude.SPLIT_BEFORE, fp.DelimiterAction.STOP_PARSING)},
            "Testing # 123",
            ((0, 0, 0, 8, "Testing "),),
        ),
    ],
)
def test_split_single_line(delimiters, content, expected):
    out = tuple(fp.Spliter(content, delimiters))
    assert out == expected


@pytest.mark.parametrize(
    "delimiters,content,expected",
    [
        # ### 0
        (
            {},
            "Testing # 123\nCaption # 456",
            ((0, 0, 1, 13, "Testing # 123\nCaption # 456"),),
        ),
        # ### 1
        (
            {"#": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.STOP_PARSING)},
            "Testing # 123\nCaption # 456",
            ((0, 0, 0, 8, "Testing "),),
        ),
        # ### 2
        (
            {"#": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.STOP_PARSING_LINE)},
            "Testing # 123\nCaption # 456",
            (
                (0, 0, 0, 8, "Testing "),
                (1, 0, 1, 8, "Caption "),
            ),
        ),
        # ### 3
        (
            {"#": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL)},
            "Testing # 123\nCaption # 456",
            (
                (0, 0, 0, 8, "Testing "),
                (0, 9, 1, 8, " 123\nCaption "),
                (1, 9, 1, 13, " 456"),
            ),
        ),
        # ### 4
        (
            {
                "#": (
                    fp.DelimiterInclude.SPLIT_BEFORE,
                    fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
                ),
                "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
            },
            "Testing # 123\nCaption # 456",
            (
                (0, 0, 0, 8, "Testing "),
                (0, 8, 0, 13, "# 123"),
                (1, 0, 1, 8, "Caption "),
                (1, 8, 1, 13, "# 456"),
            ),
        ),
        # ### 5
        (
            {
                "#": (
                    fp.DelimiterInclude.SPLIT_BEFORE,
                    fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
                ),
                "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
            },
            "Testing ## 123\nCaption ## 456",
            (
                (0, 0, 0, 8, "Testing "),
                (0, 8, 0, 14, "## 123"),
                (1, 0, 1, 8, "Caption "),
                (1, 8, 1, 14, "## 456"),
            ),
        ),
        # ### 6
        (
            {
                "#": (fp.DelimiterInclude.SPLIT_BEFORE, fp.DelimiterAction.CONTINUE),
                "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
            },
            "Testing ## 123\nCaption ## 456",
            (
                (0, 0, 0, 8, "Testing "),
                (0, 8, 0, 9, "#"),
                (0, 9, 0, 14, "# 123"),
                (1, 0, 1, 8, "Caption "),
                (1, 8, 1, 9, "#"),
                (1, 9, 1, 14, "# 456"),
            ),
        ),
        # ### 7
        (
            {
                "#": (
                    fp.DelimiterInclude.SPLIT_BEFORE,
                    fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
                ),
                "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
                "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
            },
            "Testing ## 123\nCaption ## 456",
            (
                (0, 0, 0, 8, "Testing "),
                (0, 8, 0, 14, "## 123"),
                (1, 0, 1, 8, "Caption "),
                (1, 8, 1, 14, "## 456"),
            ),
        ),
    ],
)
def test_split_multi_line(delimiters, content, expected):
    out = tuple(fp.Spliter(content, delimiters))
    assert out == expected


def test_statement():
    dlm = {
        "#": (
            fp.DelimiterInclude.SPLIT_BEFORE,
            fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
        ),
        "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
    }
    content = "Testing ## 123\nCaption ## 456"
    bi = fp.StatementIterator(content, dlm)
    assert bi.peek().raw_strip == "Testing"
    assert next(bi).raw_strip == "Testing"
    assert bi.peek().raw_strip == "## 123"
    assert next(bi).raw_strip == "## 123"

    el = next(bi)
    # strip spaces now changes the element
    # not the parser.
    assert el.raw == "Caption"
    assert el.raw_strip == "Caption"
    assert el.start_line == 2
    assert el.start_col == 0
    assert el.end_line == 2
    assert el.end_col == 7

    assert next(bi).raw_strip == "## 456"
    assert bi.peek("blip") == "blip"
    with pytest.raises(StopIteration):
        bi.peek()
    with pytest.raises(StopIteration):
        next(bi)


def test_statement2():
    dlm = {
        "#": (
            fp.DelimiterInclude.SPLIT_BEFORE,
            fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
        ),
        "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
    }
    content = "Testing ## 123\nCaption ## 456"
    bi = fp.StatementIterator(content, dlm)
    assert bi.peek().raw_strip == "Testing"
    assert next(bi).raw_strip == "Testing"
    assert bi.peek().raw_strip == "## 123"
    assert next(bi).raw_strip == "## 123"

    el = next(bi)
    # strip spaces now changes the element
    # not the parser.
    assert el.raw == "Caption"
    assert el.raw_strip == "Caption"
    assert el.start_line == 2
    assert el.start_col == 0
    assert el.end_line == 2
    assert el.end_col == 7

    assert next(bi).raw_strip == "## 456"
    assert bi.peek("blip") == "blip"
    with pytest.raises(StopIteration):
        bi.peek()
    with pytest.raises(StopIteration):
        next(bi)


def test_statement_change_dlm():
    dlm = {
        "#": (
            fp.DelimiterInclude.SPLIT_BEFORE,
            fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
        ),
        "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
    }

    dlm_new = {
        "!": (
            fp.DelimiterInclude.SPLIT_BEFORE,
            fp.DelimiterAction.CAPTURE_NEXT_TIL_EOL,
        ),
        "\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
        "\r\n": (fp.DelimiterInclude.SPLIT, fp.DelimiterAction.CONTINUE),
    }
    content = "Testing ## 123\nCaption !! 456"
    bi = fp.StatementIterator(content, dlm)
    assert bi.peek().raw_strip == "Testing"
    assert next(bi).raw_strip == "Testing"
    assert bi.peek().raw_strip == "## 123"
    assert next(bi).raw_strip == "## 123"

    assert bi.peek().raw_strip == "Caption !! 456"
    bi.set_delimiters(dlm_new)
    assert bi.peek().raw_strip == "Caption"
    assert next(bi).raw_strip == "Caption"
    assert next(bi).raw_strip == "!! 456"
