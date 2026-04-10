"""Online cache for prompt embeddings using TensorDict memory-mapped storage.

See: https://docs.pytorch.org/tensordict/main/saving.html
"""

from __future__ import annotations

from tensordict_cache._version import __version__  # noqa: F401

import hashlib
import os
import shutil

from tensordict import TensorDict


def _key_to_basename(key: str | int) -> str:
    """Convert a cache key to a filesystem-safe basename (no extension)."""
    key_str = str(key).encode("utf-8")
    return hashlib.sha256(key_str).hexdigest()[:32]


def _load_cache_from_disk(prefix: str) -> TensorDict:
    """Discover subdirs under prefix and load each via TensorDict.load_memmap."""
    cache = TensorDict({}, batch_size=[])
    if not os.path.isdir(prefix):
        return cache
    for name in os.listdir(prefix):
        subdir = os.path.join(prefix, name)
        if os.path.isdir(subdir):
            try:
                cache[name] = TensorDict.load_memmap(subdir)
            except Exception:
                pass
    return cache


def _get_dir_size(path: str) -> int:
    """Return total file size in bytes under a directory, recursively."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


class TensorCache:
    """Online cache for prompt embeddings.

    Uses TensorDict's memory-mapped storage so that each entry is stored in a
    subdirectory ``prefix/{key}/`` via :meth:`TensorDict.memmap`. TensorDict
    manages its own metadata on disk. The cache is reopened by scanning subdirs
    and loading each with :meth:`TensorDict.load_memmap`.
    """

    def __init__(
        self,
        prefix: str | os.PathLike,
        load_existing: bool = True,
        max_size_bytes: int | None = None,
    ):
        """Create or open an online cache under ``prefix``.

        Args:
            prefix: Directory where each cached TensorDict is stored in a
                subdirectory. Created if it does not exist.
            load_existing: If True, discover existing subdirs under ``prefix``
                and load each via :meth:`TensorDict.load_memmap`.
            max_size_bytes: Maximum total cache size in bytes. When exceeded,
                the oldest entries (by directory mtime) are evicted. If None,
                the cache grows without limit.
        """
        self._prefix = os.fspath(prefix)
        self._max_size_bytes = max_size_bytes
        os.makedirs(self._prefix, exist_ok=True)
        if load_existing:
            self._cache = _load_cache_from_disk(self._prefix)
        else:
            self._cache = TensorDict({}, batch_size=[])

    def key_to_basename(self, key: str | int) -> str:
        """Convert a cache key to a filesystem-safe basename (no extension)."""
        return _key_to_basename(key)

    def __setitem__(self, key: str | int, value: TensorDict) -> None:
        """Store a TensorDict in the cache under ``key``.

        The TensorDict is written to a subdirectory ``prefix/{key}/`` via
        :meth:`TensorDict.memmap`; TensorDict manages its own metadata there.
        """
        base = _key_to_basename(key)
        subdir = os.path.join(self._prefix, base)
        os.makedirs(subdir, exist_ok=True)
        mm_td = value.memmap(subdir, robust_key=True)
        self._cache[base] = mm_td
        self._evict_if_needed()

    def __getitem__(self, key: str | int) -> TensorDict:
        """Return the cached TensorDict for ``key``."""
        base = _key_to_basename(key)
        if base not in self._cache.keys():
            raise KeyError(key)
        return self._cache[base]

    def __delitem__(self, key: str | int) -> None:
        """Remove the cached entry for ``key`` from memory and disk.

        Raises:
            KeyError: If ``key`` is not in the cache.
        """
        base = _key_to_basename(key)
        if base not in self._cache.keys():
            raise KeyError(key)
        subdir = os.path.join(self._prefix, base)
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)
        del self._cache[base]

    def __contains__(self, key: str | int) -> bool:
        """Return whether ``key`` is in the cache."""
        return _key_to_basename(key) in self._cache.keys()

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache.keys())

    def keys(self) -> list[str]:
        """Return the list of cache key basenames (hashed names)."""
        return list(self._cache.keys())

    def get(self, key: str | int, default: TensorDict | None = None) -> TensorDict | None:
        """Return the cached TensorDict for ``key``, or ``default`` if missing."""
        try:
            return self[key]
        except KeyError:
            return default

    def _evict_if_needed(self) -> None:
        """Evict oldest entries (by directory mtime) until cache is within size limit."""
        if self._max_size_bytes is None:
            return
        while self.get_cache_size() > self._max_size_bytes and len(self) > 0:
            oldest_name = None
            oldest_mtime = float("inf")
            for name in list(self._cache.keys()):
                subdir = os.path.join(self._prefix, name)
                if os.path.isdir(subdir):
                    mtime = os.path.getmtime(subdir)
                    if mtime < oldest_mtime:
                        oldest_mtime = mtime
                        oldest_name = name
            if oldest_name is None:
                break
            subdir = os.path.join(self._prefix, oldest_name)
            if os.path.isdir(subdir):
                shutil.rmtree(subdir)
            del self._cache[oldest_name]

    def clear(self) -> None:
        """Clear the cache and remove all cached subdirectories on disk."""
        for name in list(self._cache.keys()):
            subdir = os.path.join(self._prefix, name)
            if os.path.isdir(subdir):
                shutil.rmtree(subdir)
        self._cache = TensorDict({}, batch_size=[])

    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        return f"TensorCache(prefix={self._prefix}, n_cache={len(self._cache.keys())})"

    def __str__(self) -> str:
        """Return a string representation of the cache."""
        return self.__repr__()

    def get_cache_size(self) -> int:
        """Return the total size of the cache in bytes."""
        return sum(
            _get_dir_size(os.path.join(self._prefix, name))
            for name in self._cache.keys()
        )

    def get_cache_size_human(self) -> str:
        """Return the size of the cache in a human-readable format."""
        return format_big_number(self.get_cache_size())

