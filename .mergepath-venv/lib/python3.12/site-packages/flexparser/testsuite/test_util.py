from __future__ import annotations

import hashlib
import typing

import pytest

import flexparser.flexparser as fp
from flexparser.flexparser import _HASH_ALGORITHMS


def test_yield_types():
    class X:
        pass

    assert tuple(fp._yield_types(float)) == (float,)
    assert tuple(fp._yield_types(X)) == (X,)
    assert tuple(fp._yield_types(X())) == ()


def test_yield_types_container():
    class X:
        pass

    o = tuple[float, X]
    assert tuple(fp._yield_types(o)) == (float, X)

    o = tuple[float, ...]
    assert tuple(fp._yield_types(o)) == (float,)

    o = tuple[typing.Union[float, X], ...]
    assert tuple(fp._yield_types(o)) == (float, X)


def test_yield_types_union():
    class X:
        pass

    o = typing.Union[float, X]
    assert tuple(fp._yield_types(o)) == (float, X)


def test_yield_types_list():
    o = list[float]
    assert tuple(fp._yield_types(o)) == (float,)


def test_hash_object():
    content = b"spam \n ham"
    hasher = hashlib.sha1

    ho = fp.Hash.from_bytes(hashlib.sha1, content)
    hd = hasher(content).hexdigest()
    assert ho.algorithm_name == "sha1"
    assert ho.hexdigest == hd
    assert ho != hd
    assert ho != fp.Hash.from_bytes(hashlib.md5, content)
    assert ho == fp.Hash.from_bytes(hashlib.sha1, content)


@pytest.mark.parametrize("algo_name", _HASH_ALGORITHMS)
def test_hash_items(algo_name: str):
    content = b"spam \n ham"
    hasher = getattr(hashlib, algo_name)

    ho = fp.Hash.from_bytes(hasher, content)
    hd = hasher(content).hexdigest()
    assert ho.algorithm_name == algo_name
    assert ho.hexdigest == hd
    assert ho != hd
    assert ho == fp.Hash.from_bytes(hasher, content)
