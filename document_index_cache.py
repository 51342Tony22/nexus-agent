# Small in-process LRU cache for reusable document indexes.

from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Callable


class DocumentIndexCache:
    def __init__(self, max_entries: int = 6):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")

        self.max_entries = max_entries
        self._entries = OrderedDict()
        self.build_count = 0
        self.eviction_count = 0
        self.invalidation_count = 0

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def get_or_build(
        self,
        path: Path,
        index_signature: tuple,
        builder: Callable[[], object],
    ) -> tuple[object, dict]:
        resolved_path = path.resolve()
        file_size, modified_ns = self._fingerprint(resolved_path)
        cache_key = (
            str(resolved_path),
            file_size,
            modified_ns,
            index_signature,
        )

        if cache_key in self._entries:
            value = self._entries.pop(cache_key)
            self._entries[cache_key] = value
            return value, self._debug_info(
                hit=True,
                file_size=file_size,
                modified_ns=modified_ns,
                build_seconds=0.0,
            )

        # Remove an older index for the same path when the file fingerprint or
        # indexing settings have changed.
        stale_keys = [
            key
            for key in self._entries
            if key[0] == str(resolved_path)
        ]
        for stale_key in stale_keys:
            del self._entries[stale_key]
            self.invalidation_count += 1

        started_at = perf_counter()
        value = builder()
        build_seconds = perf_counter() - started_at
        self.build_count += 1
        self._entries[cache_key] = value

        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.eviction_count += 1

        return value, self._debug_info(
            hit=False,
            file_size=file_size,
            modified_ns=modified_ns,
            build_seconds=build_seconds,
        )

    def _debug_info(
        self,
        hit: bool,
        file_size: int,
        modified_ns: int,
        build_seconds: float,
    ) -> dict:
        return {
            "hit": hit,
            "cache_size": len(self._entries),
            "max_entries": self.max_entries,
            "build_count": self.build_count,
            "eviction_count": self.eviction_count,
            "invalidation_count": self.invalidation_count,
            "build_seconds": build_seconds,
            "file_size_bytes": file_size,
            "file_modified_ns": modified_ns,
        }

    def clear(self, reset_stats: bool = True):
        self._entries.clear()

        if reset_stats:
            self.build_count = 0
            self.eviction_count = 0
            self.invalidation_count = 0

    def __len__(self) -> int:
        return len(self._entries)
