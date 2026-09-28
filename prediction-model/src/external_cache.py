"""
External Cache — Append-Only Source Data Cache.

Manages raw external source data files with:
  - append-only storage (no silent replacement)
  - revision tracking with new object IDs
  - file integrity verification
  - version-aware selection
  - production reads from verified files only
"""

import os
import json
import hashlib
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


class CacheError(Exception):
    """Raised when a cache operation fails."""
    pass


class ExternalCache:
    """
    Append-only cache for raw external source data.

    Rules:
      - Raw data is append-only.
      - Raw files are never silently replaced.
      - Revisions receive new object IDs.
      - Production inference reads prepared, verified files only.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.abspath(cache_dir)
        self._index_path = os.path.join(self.cache_dir, "cache_index.json")
        self._index: Dict[str, Any] = {"objects": {}, "version": "1.0.0"}
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        """Load the cache index, creating it if it does not exist."""
        if os.path.exists(self._index_path):
            with open(self._index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        else:
            self._save_index()

    def _save_index(self) -> None:
        """Persist the cache index."""
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)

    @staticmethod
    def compute_file_sha256(path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def store(
        self,
        object_id: str,
        source_id: str,
        data_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a raw data object in the cache.

        Append-only: if object_id already exists, raises CacheError.
        Revisions must use a distinct object_id.
        """
        if object_id in self._index.get("objects", {}):
            raise CacheError(
                f"Object '{object_id}' already exists in cache. "
                "Raw files are never silently replaced. "
                "Use a new object_id for revisions."
            )

        # Create source subdirectory
        source_dir = os.path.join(self.cache_dir, source_id)
        os.makedirs(source_dir, exist_ok=True)

        # Write raw data
        raw_path = os.path.join(source_dir, f"{object_id}.raw")
        with open(raw_path, "wb") as f:
            f.write(data_bytes)

        # Compute checksum
        checksum = hashlib.sha256(data_bytes).hexdigest()

        # Build index record
        record = {
            "object_id": object_id,
            "source_id": source_id,
            "raw_path": raw_path,
            "checksum_sha256": checksum,
            "size_bytes": len(data_bytes),
            "stored_at_utc": datetime.now(timezone.utc).isoformat(),
            "verified": False,
            "metadata": metadata or {},
        }

        self._index.setdefault("objects", {})[object_id] = record
        self._save_index()

        return record

    def verify(self, object_id: str) -> bool:
        """
        Verify a cached object's integrity against its stored checksum.

        Marks the object as verified if the check passes.
        """
        if object_id not in self._index.get("objects", {}):
            raise CacheError(f"Object '{object_id}' not found in cache")

        record = self._index["objects"][object_id]
        raw_path = record["raw_path"]

        if not os.path.exists(raw_path):
            raise CacheError(f"Raw file missing: {raw_path}")

        actual_checksum = self.compute_file_sha256(raw_path)
        expected_checksum = record["checksum_sha256"]

        if actual_checksum != expected_checksum:
            record["verified"] = False
            self._save_index()
            raise CacheError(
                f"Checksum mismatch for '{object_id}': "
                f"expected {expected_checksum[:16]}..., "
                f"got {actual_checksum[:16]}..."
            )

        record["verified"] = True
        record["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return True

    def read_verified(self, object_id: str) -> bytes:
        """
        Read a cached object only if it has been verified.

        Production inference must use this method.
        """
        if object_id not in self._index.get("objects", {}):
            raise CacheError(f"Object '{object_id}' not found in cache")

        record = self._index["objects"][object_id]
        if not record.get("verified", False):
            raise CacheError(
                f"Object '{object_id}' has not been verified. "
                "Production reads require verified objects."
            )

        raw_path = record["raw_path"]
        if not os.path.exists(raw_path):
            raise CacheError(f"Raw file missing: {raw_path}")

        with open(raw_path, "rb") as f:
            return f.read()

    def get_record(self, object_id: str) -> Dict[str, Any]:
        """Get the index record for a cached object."""
        if object_id not in self._index.get("objects", {}):
            raise CacheError(f"Object '{object_id}' not found in cache")
        return dict(self._index["objects"][object_id])

    def list_objects(self, source_id: Optional[str] = None) -> List[str]:
        """List cached object IDs, optionally filtered by source."""
        objects = self._index.get("objects", {})
        if source_id:
            return [
                oid for oid, rec in objects.items()
                if rec.get("source_id") == source_id
            ]
        return list(objects.keys())

    def list_verified_objects(self, source_id: Optional[str] = None) -> List[str]:
        """List only verified object IDs."""
        objects = self._index.get("objects", {})
        result = []
        for oid, rec in objects.items():
            if rec.get("verified", False):
                if source_id is None or rec.get("source_id") == source_id:
                    result.append(oid)
        return result

    @property
    def object_count(self) -> int:
        """Total number of cached objects."""
        return len(self._index.get("objects", {}))

    @property
    def cache_index_hash(self) -> str:
        """SHA-256 of the cache index for provenance."""
        raw = json.dumps(self._index, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
