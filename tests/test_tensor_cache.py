from __future__ import annotations

import hashlib
import os
import time

import pytest
import torch
from tensordict import TensorDict

from tensordict_cache import (
    TensorCache,
    _get_dir_size,
    _key_to_basename,
    _load_cache_from_disk,
    format_big_number,
)


# ---------------------------------------------------------------------------
# Helper: _key_to_basename
# ---------------------------------------------------------------------------


class TestKeyToBasename:
    def test_returns_hex_string(self):
        result = _key_to_basename("hello")
        assert isinstance(result, str)
        assert len(result) == 32
        # Should be valid hex
        int(result, 16)

    def test_deterministic(self):
        assert _key_to_basename("foo") == _key_to_basename("foo")

    def test_different_keys_different_basenames(self):
        assert _key_to_basename("a") != _key_to_basename("b")

    def test_int_key(self):
        result = _key_to_basename(42)
        expected = hashlib.sha256(b"42").hexdigest()[:32]
        assert result == expected

    def test_string_key(self):
        result = _key_to_basename("test")
        expected = hashlib.sha256(b"test").hexdigest()[:32]
        assert result == expected


# ---------------------------------------------------------------------------
# Helper: _load_cache_from_disk
# ---------------------------------------------------------------------------


class TestLoadCacheFromDisk:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        cache = _load_cache_from_disk(str(tmp_path / "nonexistent"))
        assert len(cache.keys()) == 0

    def test_empty_dir_returns_empty(self, tmp_path):
        cache = _load_cache_from_disk(str(tmp_path))
        assert len(cache.keys()) == 0

    def test_loads_existing_memmap(self, tmp_path):
        td = TensorDict({"x": torch.tensor([1.0, 2.0])}, batch_size=[])
        subdir = str(tmp_path / "entry1")
        td.memmap(subdir)

        cache = _load_cache_from_disk(str(tmp_path))
        assert "entry1" in cache.keys()
        assert torch.equal(cache["entry1"]["x"], torch.tensor([1.0, 2.0]))

    def test_skips_non_tensordict_subdirs(self, tmp_path):
        # Create a subdir that isn't a valid memmap
        bad_dir = tmp_path / "bad_entry"
        bad_dir.mkdir()
        (bad_dir / "random.txt").write_text("not a tensordict")

        cache = _load_cache_from_disk(str(tmp_path))
        assert "bad_entry" not in cache.keys()

    def test_skips_files(self, tmp_path):
        (tmp_path / "somefile.txt").write_text("hello")
        cache = _load_cache_from_disk(str(tmp_path))
        assert len(cache.keys()) == 0


# ---------------------------------------------------------------------------
# Helper: format_big_number
# ---------------------------------------------------------------------------


class TestFormatBigNumber:
    def test_small_number(self):
        assert format_big_number(500) == "500"

    def test_thousands(self):
        assert format_big_number(1500) == "2K"

    def test_millions(self):
        assert format_big_number(2_500_000) == "2M"

    def test_precision(self):
        assert format_big_number(1500, precision=1) == "1.5K"

    def test_zero(self):
        assert format_big_number(0) == "0"

    def test_negative(self):
        assert format_big_number(-500) == "-500"

    def test_very_large_number(self):
        # Beyond quadrillions, returns raw divided number
        result = format_big_number(10**18)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TensorCache
# ---------------------------------------------------------------------------


class TestTensorCacheInit:
    def test_creates_prefix_dir(self, tmp_path):
        prefix = tmp_path / "cache_dir"
        assert not prefix.exists()
        TensorCache(str(prefix))
        assert prefix.is_dir()

    def test_accepts_pathlike(self, tmp_path):
        prefix = tmp_path / "cache_dir"
        cache = TensorCache(prefix)
        assert len(cache) == 0

    def test_empty_cache_on_init(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert len(cache) == 0

    def test_load_existing_false(self, tmp_path):
        # Store something first
        cache1 = TensorCache(str(tmp_path))
        cache1["key1"] = TensorDict({"v": torch.tensor([1.0])}, batch_size=[])

        # Reopen with load_existing=False
        cache2 = TensorCache(str(tmp_path), load_existing=False)
        assert len(cache2) == 0

    def test_load_existing_true(self, tmp_path):
        cache1 = TensorCache(str(tmp_path))
        cache1["key1"] = TensorDict({"v": torch.tensor([1.0])}, batch_size=[])

        cache2 = TensorCache(str(tmp_path), load_existing=True)
        assert len(cache2) == 1
        assert "key1" in cache2


class TestTensorCacheSetGet:
    def test_set_and_get(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        td = TensorDict({"a": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[])
        cache["mykey"] = td

        result = cache["mykey"]
        assert torch.equal(result["a"], torch.tensor([1.0, 2.0, 3.0]))

    def test_set_int_key(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        td = TensorDict({"val": torch.tensor(42.0)}, batch_size=[])
        cache[0] = td

        result = cache[0]
        assert torch.equal(result["val"], torch.tensor(42.0))

    def test_get_missing_key_raises(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        with pytest.raises(KeyError):
            cache["nonexistent"]

    def test_overwrite_key(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["k"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        cache["k"] = TensorDict({"x": torch.tensor(2.0)}, batch_size=[])

        result = cache["k"]
        assert torch.equal(result["x"], torch.tensor(2.0))

    def test_multiple_keys(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        for i in range(5):
            cache[f"key_{i}"] = TensorDict(
                {"v": torch.tensor(float(i))}, batch_size=[]
            )

        assert len(cache) == 5
        for i in range(5):
            assert torch.equal(cache[f"key_{i}"]["v"], torch.tensor(float(i)))


class TestTensorCacheContains:
    def test_contains_existing(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["present"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        assert "present" in cache

    def test_not_contains_missing(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert "absent" not in cache

    def test_contains_int_key(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache[99] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        assert 99 in cache
        assert 100 not in cache


class TestTensorCacheLen:
    def test_empty(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert len(cache) == 0

    def test_after_inserts(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["a"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        cache["b"] = TensorDict({"x": torch.tensor(2.0)}, batch_size=[])
        assert len(cache) == 2


class TestTensorCacheKeys:
    def test_keys_empty(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert cache.keys() == []

    def test_keys_returns_hashed_names(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["hello"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])

        keys = cache.keys()
        assert len(keys) == 1
        assert keys[0] == _key_to_basename("hello")


class TestTensorCacheGet:
    def test_get_existing(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["k"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        result = cache.get("k")
        assert result is not None
        assert torch.equal(result["x"], torch.tensor(1.0))

    def test_get_missing_returns_none(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert cache.get("missing") is None

    def test_get_missing_returns_default(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        default = TensorDict({"d": torch.tensor(0.0)}, batch_size=[])
        result = cache.get("missing", default)
        assert result is default


class TestTensorCacheClear:
    def test_clear_empties_cache(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["a"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        cache["b"] = TensorDict({"x": torch.tensor(2.0)}, batch_size=[])
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert "a" not in cache

    def test_clear_removes_disk_files(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["a"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        base = _key_to_basename("a")
        subdir = tmp_path / base
        assert subdir.is_dir()

        cache.clear()
        assert not subdir.exists()

    def test_clear_on_empty_cache(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache.clear()  # Should not raise
        assert len(cache) == 0


class TestTensorCacheRepr:
    def test_repr(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        r = repr(cache)
        assert "TensorCache" in r
        assert str(tmp_path) in r
        assert "n_cache=0" in r

    def test_str_matches_repr(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert str(cache) == repr(cache)

    def test_repr_with_entries(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["a"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        assert "n_cache=1" in repr(cache)


class TestTensorCacheKeyToBasename:
    def test_instance_method_matches_module_function(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert cache.key_to_basename("test") == _key_to_basename("test")
        assert cache.key_to_basename(42) == _key_to_basename(42)


class TestTensorCachePersistence:
    def test_data_persists_across_instances(self, tmp_path):
        cache1 = TensorCache(str(tmp_path))
        td = TensorDict(
            {"embedding": torch.randn(128)},
            batch_size=[],
        )
        cache1["prompt_1"] = td
        original_embedding = cache1["prompt_1"]["embedding"].clone()

        # New instance loads from disk
        cache2 = TensorCache(str(tmp_path))
        assert "prompt_1" in cache2
        assert torch.equal(cache2["prompt_1"]["embedding"], original_embedding)

    def test_multi_tensor_value(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        td = TensorDict(
            {
                "hidden": torch.randn(64),
                "logits": torch.randn(10),
            },
            batch_size=[],
        )
        cache["multi"] = td

        result = cache["multi"]
        assert "hidden" in result.keys()
        assert "logits" in result.keys()
        assert result["hidden"].shape == (64,)
        assert result["logits"].shape == (10,)


# ---------------------------------------------------------------------------
# TensorCache: __delitem__
# ---------------------------------------------------------------------------


class TestTensorCacheDelItem:
    def test_delitem_removes_from_cache(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["k"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        del cache["k"]
        assert "k" not in cache
        assert len(cache) == 0

    def test_delitem_removes_disk_files(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["k"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        subdir = tmp_path / _key_to_basename("k")
        assert subdir.is_dir()

        del cache["k"]
        assert not subdir.exists()

    def test_delitem_missing_key_raises(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        with pytest.raises(KeyError):
            del cache["nonexistent"]

    def test_delitem_int_key(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache[42] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        del cache[42]
        assert 42 not in cache

    def test_delitem_does_not_affect_other_entries(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["a"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        cache["b"] = TensorDict({"x": torch.tensor(2.0)}, batch_size=[])
        del cache["a"]

        assert "a" not in cache
        assert "b" in cache
        assert torch.equal(cache["b"]["x"], torch.tensor(2.0))

    def test_delitem_then_reinsert(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["k"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        del cache["k"]
        cache["k"] = TensorDict({"x": torch.tensor(99.0)}, batch_size=[])
        assert torch.equal(cache["k"]["x"], torch.tensor(99.0))

    def test_delitem_persistence(self, tmp_path):
        cache1 = TensorCache(str(tmp_path))
        cache1["a"] = TensorDict({"x": torch.tensor(1.0)}, batch_size=[])
        cache1["b"] = TensorDict({"x": torch.tensor(2.0)}, batch_size=[])
        del cache1["a"]

        cache2 = TensorCache(str(tmp_path))
        assert "a" not in cache2
        assert "b" in cache2


# ---------------------------------------------------------------------------
# TensorCache: get_cache_size (fix verification)
# ---------------------------------------------------------------------------


class TestTensorCacheCacheSize:
    def test_cache_size_empty(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert cache.get_cache_size() == 0

    def test_cache_size_positive_after_insert(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["k"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        assert cache.get_cache_size() > 0

    def test_cache_size_increases_with_entries(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["a"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        size1 = cache.get_cache_size()
        cache["b"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        size2 = cache.get_cache_size()
        assert size2 > size1

    def test_cache_size_decreases_after_delete(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["a"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        cache["b"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        size_before = cache.get_cache_size()
        del cache["a"]
        size_after = cache.get_cache_size()
        assert size_after < size_before


# ---------------------------------------------------------------------------
# Helper: _get_dir_size
# ---------------------------------------------------------------------------


class TestGetDirSize:
    def test_empty_dir(self, tmp_path):
        assert _get_dir_size(str(tmp_path)) == 0

    def test_dir_with_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        size = _get_dir_size(str(tmp_path))
        assert size == 10  # 5 + 5 bytes

    def test_nested_dirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("data")
        size = _get_dir_size(str(tmp_path))
        assert size == 4


# ---------------------------------------------------------------------------
# TensorCache: max_size_bytes and eviction
# ---------------------------------------------------------------------------


class TestTensorCacheEviction:
    def test_init_accepts_max_size_bytes(self, tmp_path):
        cache = TensorCache(str(tmp_path), max_size_bytes=10000)
        assert cache._max_size_bytes == 10000

    def test_init_default_max_size_bytes_is_none(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        assert cache._max_size_bytes is None

    def test_no_eviction_when_under_limit(self, tmp_path):
        cache = TensorCache(str(tmp_path), max_size_bytes=10_000_000)
        for i in range(5):
            cache[f"k{i}"] = TensorDict({"x": torch.randn(16)}, batch_size=[])
        assert len(cache) == 5

    def test_eviction_when_exceeding_limit(self, tmp_path):
        # Insert one entry to measure its size
        cache = TensorCache(str(tmp_path))
        cache["measure"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        entry_size = cache.get_cache_size()
        cache.clear()

        # Set limit to ~2.5 entries
        limit = int(entry_size * 2.5)
        cache = TensorCache(str(tmp_path), max_size_bytes=limit)
        for i in range(4):
            cache[f"k{i}"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
            time.sleep(0.05)  # Ensure different mtimes

        # Should have evicted older entries to stay within limit
        assert len(cache) <= 3
        assert cache.get_cache_size() <= limit

    def test_eviction_removes_oldest_first(self, tmp_path):
        cache = TensorCache(str(tmp_path))

        # Insert entries with clear time ordering
        cache["old"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        old_base = _key_to_basename("old")
        old_subdir = os.path.join(str(tmp_path), old_base)
        # Set old mtime explicitly
        os.utime(old_subdir, (1000, 1000))

        time.sleep(0.05)
        cache["new"] = TensorDict({"x": torch.randn(128)}, batch_size=[])

        entry_size = _get_dir_size(
            os.path.join(str(tmp_path), _key_to_basename("new"))
        )

        # Now create a size-limited cache that can only hold ~1.5 entries
        limit = int(entry_size * 1.5)
        cache2 = TensorCache(str(tmp_path), max_size_bytes=limit)

        # Trigger eviction by inserting one more
        cache2["newest"] = TensorDict({"x": torch.randn(128)}, batch_size=[])

        # "old" should be evicted (oldest mtime), "new" and/or "newest" should remain
        assert not os.path.isdir(old_subdir)

    def test_eviction_removes_disk_files(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["first"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        first_subdir = tmp_path / _key_to_basename("first")
        # Set old mtime
        os.utime(str(first_subdir), (1000, 1000))

        entry_size = cache.get_cache_size()
        cache_limited = TensorCache(
            str(tmp_path), max_size_bytes=int(entry_size * 1.5)
        )
        cache_limited["second"] = TensorDict({"x": torch.randn(128)}, batch_size=[])

        assert not first_subdir.exists()

    def test_single_entry_exceeds_limit(self, tmp_path):
        cache = TensorCache(str(tmp_path), max_size_bytes=1)
        cache["big"] = TensorDict({"x": torch.randn(256)}, batch_size=[])
        # Entry remains since it's the only (and newest) one
        # After eviction loop, it may or may not remain depending on implementation
        # but the cache should not crash
        assert len(cache) <= 1

    def test_max_size_none_no_eviction(self, tmp_path):
        cache = TensorCache(str(tmp_path), max_size_bytes=None)
        for i in range(10):
            cache[f"k{i}"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        assert len(cache) == 10

    def test_eviction_persistence(self, tmp_path):
        cache = TensorCache(str(tmp_path))
        cache["old"] = TensorDict({"x": torch.randn(128)}, batch_size=[])
        old_subdir = str(tmp_path / _key_to_basename("old"))
        os.utime(old_subdir, (1000, 1000))

        entry_size = cache.get_cache_size()
        cache2 = TensorCache(
            str(tmp_path), max_size_bytes=int(entry_size * 1.5)
        )
        cache2["new"] = TensorDict({"x": torch.randn(128)}, batch_size=[])

        # Reopen without limit - evicted entries should stay gone
        cache3 = TensorCache(str(tmp_path))
        assert "old" not in cache3
