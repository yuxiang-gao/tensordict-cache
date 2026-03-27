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

    def __init__(self, prefix: str | os.PathLike, load_existing: bool = True):
        """Create or open an online cache under ``prefix``.

        Args:
            prefix: Directory where each cached TensorDict is stored in a
                subdirectory. Created if it does not exist.
            load_existing: If True, discover existing subdirs under ``prefix``
                and load each via :meth:`TensorDict.load_memmap`.
        """
        self._prefix = os.fspath(prefix)
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

    def __getitem__(self, key: str | int) -> TensorDict:
        """Return the cached TensorDict for ``key``."""
        base = _key_to_basename(key)
        if base not in self._cache.keys():
            raise KeyError(key)
        return self._cache[base]

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
        """Return the size of the cache in bytes."""
        return sum(os.path.getsize(os.path.join(self._prefix, key)) for key in self._cache.keys())

    def get_cache_size_human(self) -> str:
        """Return the size of the cache in a human-readable format."""
        return format_big_number(self.get_cache_size())

