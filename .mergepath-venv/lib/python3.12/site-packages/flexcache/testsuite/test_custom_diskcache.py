import pytest

import flexcache
from flexcache import DiskCache


def test_register(tmp_path):
    c = DiskCache(tmp_path)

    class Header(
        flexcache.InvalidateByExist, flexcache.NameByFields, flexcache.BaseHeader
    ):
        @classmethod
        def from_int(cls, source, converter_id):
            return cls(bytes(source), converter_id)

    c.register_header_class(int, Header.from_int)

    c.load(3)

    with pytest.raises(TypeError):
        c.load(3j)


def test_missing_cache_path(tmp_path):
    c = DiskCache(tmp_path)

    class Header(
        flexcache.InvalidateByExist, flexcache.NameByFields, flexcache.BaseHeader
    ):
        @classmethod
        def from_int(cls, source, converter_id):
            return cls(bytes(source), converter_id)

    hdr = Header("123", "456")
    assert c.rawsave(hdr, "789").stem == c.cache_stem_for(hdr)
    assert c.rawload(hdr) == "789"


def test_converter_id(tmp_path):
    c = DiskCache(tmp_path)

    class Header(
        flexcache.InvalidateByExist, flexcache.NameByFields, flexcache.BaseHeader
    ):
        @classmethod
        def from_int(cls, source, converter_id):
            return cls(bytes(source), converter_id)

    c.register_header_class(int, Header.from_int)

    def func(n):
        return n * 2

    content, this_hash = c.load(21, func)
    assert content == 42
    assert c.load(21, "func") == (content, this_hash)
    assert c.save(content, 21, "func") == this_hash


def test_converter_pass_hash(tmp_path):
    c = DiskCache(tmp_path)

    class Header(
        flexcache.InvalidateByExist, flexcache.NameByFields, flexcache.BaseHeader
    ):
        @classmethod
        def from_int(cls, source, converter_id):
            return cls(bytes(source), converter_id)

    c.register_header_class(int, Header.from_int)

    def func(n, a_hash):
        return (n, a_hash)

    content, this_hash = c.load(21, func, True)
    assert content == (21, this_hash)
    assert c.load(21, "func", True) == (content, this_hash)
    assert c.save(content, 21, "func") == this_hash
