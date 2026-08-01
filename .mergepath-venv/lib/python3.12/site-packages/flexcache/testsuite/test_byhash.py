import pathlib
import pickle
import time

from flexcache import DiskCacheByHash

# These sleep time is needed when run on GitHub Actions
# If not given or too short, some mtime changes are not visible.
FS_SLEEP = 0.010


def parser(p: pathlib.Path):
    return p.read_bytes()


def test_file_changed(tmp_path):
    # Generate a definition file
    dfile = tmp_path / "definitions.txt"
    dfile.write_bytes(b"1234")

    dc = DiskCacheByHash(tmp_path)
    content = dc.load(dfile)[0]

    assert len(tuple(tmp_path.glob("*.pickle"))) == 0
    assert len(tuple(tmp_path.glob("*.json"))) == 0

    time.sleep(FS_SLEEP)

    # First, the cache should be missed
    assert content is None
    dc.save(pickle.dumps(dfile.read_bytes()), dfile)

    # There should be a cache file now
    assert len(tuple(tmp_path.glob("*.pickle"))) == 1
    assert len(tuple(tmp_path.glob("*.json"))) == 1

    content = pickle.loads(dc.load(dfile)[0])
    # Now, the cache should be hit
    assert content == b"1234"

    # Modify the definition file
    # Add some sleep to make sure that the time stamp difference is significant.
    dfile.write_bytes(b"1235")

    # Verify that the cache was not loaded as the content of the original file
    # has changed.
    assert dc.load(dfile)[0] is None


def test_cache_miss(tmp_path):
    # Generate a definition file
    dfile = tmp_path / "definitions.txt"
    dfile.write_bytes(b"1234")

    dc = DiskCacheByHash(tmp_path)
    content = dc.load(dfile)[0]

    assert len(tuple(tmp_path.glob("*.pickle"))) == 0
    assert len(tuple(tmp_path.glob("*.json"))) == 0

    time.sleep(FS_SLEEP)

    # First, the cache should be missed
    assert content is None
    dc.save(pickle.dumps(dfile.read_bytes()), dfile)

    # There should be a cache file now
    assert len(tuple(tmp_path.glob("*.pickle"))) == 1
    assert len(tuple(tmp_path.glob("*.json"))) == 1

    content = pickle.loads(dc.load(dfile)[0])
    # Now, the cache should be hit
    assert content == b"1234"

    # Modify the definition file
    # Add some sleep to make sure that the time stamp difference is significant.
    time.sleep(FS_SLEEP)
    dfile.write_bytes(b"1235")

    # Verify that the cached was not loaded
    content = dc.load(dfile)[0]
    assert content is None


def test_func(tmp_path):
    # Generate a definition file
    dfile = tmp_path / "definitions.txt"
    dfile.write_bytes(b"1234")

    dc = DiskCacheByHash(tmp_path)
    assert dc.load(dfile, converter=parser)[0] == b"1234"
    # There should be a cache file now
    assert len(tuple(tmp_path.glob("*.pickle"))) == 1
    assert len(tuple(tmp_path.glob("*.json"))) == 1

    # Modify the definition file
    # Add some sleep to make sure that the time stamp difference is significant.
    dfile.write_bytes(b"1235")

    # Verify that the cache was not loaded as the content of the original file
    # has changed.
    assert dc.load(dfile, converter=parser)[0] == b"1235"
    # There should be TWO cache files now
    assert len(tuple(tmp_path.glob("*.pickle"))) == 2
    assert len(tuple(tmp_path.glob("*.json"))) == 2
