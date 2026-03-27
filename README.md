# tensordict-cache

A persistent, memory-mapped cache for tensor data built on [TensorDict](https://github.com/pytorch/tensordict).

Store and retrieve `TensorDict` objects on disk using memory-mapped files. Cached entries survive process restarts and are loaded lazily without copying data into RAM.

## Installation

```bash
pip install tensordict-cache
```

## Quick start

```python
import torch
from tensordict import TensorDict
from tensordict_cache import TensorCache

# Create a cache (directory is created if it doesn't exist)
cache = TensorCache("./my_cache")

# Store embeddings keyed by prompt text
cache["hello world"] = TensorDict(
    {"embedding": torch.randn(768)},
    batch_size=[],
)

# Retrieve them
embedding = cache["hello world"]["embedding"]
```

## Usage

### Creating a cache

```python
from tensordict_cache import TensorCache

# Open or create a cache directory
cache = TensorCache("/path/to/cache")

# Open without loading existing entries
cache = TensorCache("/path/to/cache", load_existing=False)
```

### Storing entries

Keys can be strings or integers. Values must be `TensorDict` instances.

```python
import torch
from tensordict import TensorDict

td = TensorDict({
    "hidden_state": torch.randn(512),
    "logits": torch.randn(10),
}, batch_size=[])

cache["my_prompt"] = td
cache[42] = td  # integer keys work too
```

### Retrieving entries

```python
# Dict-style access (raises KeyError if missing)
result = cache["my_prompt"]

# Safe access with default
result = cache.get("my_prompt")          # returns None if missing
result = cache.get("my_prompt", default) # returns default if missing
```

### Checking membership and length

```python
if "my_prompt" in cache:
    print("Hit!")

print(f"Cache has {len(cache)} entries")
```

### Listing keys and inspecting the cache

```python
# Keys are SHA-256 hashes of the original key
print(cache.keys())

# Human-readable representation
print(cache)
# TensorCache(prefix=/path/to/cache, n_cache=3)
```

### Clearing the cache

```python
# Remove all entries from memory and disk
cache.clear()
```

### Persistence across sessions

Data is written to disk as memory-mapped files. Reopening the same directory
automatically loads all previously stored entries:

```python
# Session 1
cache = TensorCache("./my_cache")
cache["prompt_a"] = TensorDict({"v": torch.tensor(1.0)}, batch_size=[])

# Session 2 (new process)
cache = TensorCache("./my_cache")
assert "prompt_a" in cache  # still there
```

## How it works

Each entry is stored in a subdirectory under the cache prefix:

```
my_cache/
  a1b2c3d4.../   # SHA-256 hash of the key
    meta.json
    *.memmap
  e5f6g7h8.../
    meta.json
    *.memmap
```

Under the hood, `TensorCache` calls [`TensorDict.memmap()`](https://docs.pytorch.org/tensordict/main/saving.html)
to write each entry and `TensorDict.load_memmap()` to read it back. Memory-mapped
storage means tensors are not loaded into RAM until accessed, keeping memory usage
low even for large caches.

## API reference

| Method | Description |
|---|---|
| `TensorCache(prefix, load_existing=True)` | Create or open a cache at `prefix` |
| `cache[key] = td` | Store a `TensorDict` under `key` |
| `cache[key]` | Retrieve a `TensorDict` (raises `KeyError` if missing) |
| `cache.get(key, default=None)` | Retrieve or return `default` |
| `key in cache` | Check if `key` exists |
| `len(cache)` | Number of cached entries |
| `cache.keys()` | List of hashed key names |
| `cache.clear()` | Remove all entries from memory and disk |
| `cache.get_cache_size()` | Total cache size in bytes |
| `cache.get_cache_size_human()` | Cache size as human-readable string |
| `cache.key_to_basename(key)` | Get the hashed filename for a key |

## Requirements

- Python >= 3.10
- [tensordict](https://github.com/pytorch/tensordict) >= 0.11.0

## License

MIT
