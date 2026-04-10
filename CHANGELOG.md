# Changelog

## v0.2.1

### Added

- **Delete by key** — `del cache[key]` removes a single entry from memory and disk.
- **Size-limited cache** — new `max_size_bytes` parameter on `TensorCache` caps total disk usage. When exceeded, the oldest entries (by modification time) are evicted automatically.

### Fixed

- `get_cache_size()` now correctly sums file sizes recursively instead of returning directory entry sizes.

## v0.1.0

- Initial release with `TensorCache`, memory-mapped storage, and persistence across sessions.
