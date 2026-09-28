"""
External Source Contract — Immutable Source Manifests.

Defines the contract for external source data objects, ensuring
provenance, reproducibility, and causal correctness for all
external data used in training or inference.

Each source object must carry:
  - source_id and product version
  - valid_time_utc and issue_time_utc (if forecast)
  - retrieval_time_utc
  - request parameters
  - response checksum (SHA-256)
  - raw object path
  - license decision and registry hash
  - quality flags
  - latency
  - revision status
  - parser version
"""

import os
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List


# ─── Source Object Schema ─────────────────────────────────────────────

REQUIRED_OBJECT_FIELDS = [
    "source_id",
    "product_version",
    "valid_time_utc",
    "retrieval_time_utc",
    "request_parameters",
    "response_checksum_sha256",
    "raw_object_path",
    "license_decision",
    "license_registry_hash",
    "quality_flags",
    "latency_seconds",
    "revision_status",
    "parser_version",
]

OPTIONAL_OBJECT_FIELDS = [
    "issue_time_utc",  # Required for forecast data
]

VALID_REVISION_STATUSES = frozenset([
    "original",
    "revised",
    "preliminary",
    "final",
])

VALID_QUALITY_FLAGS = frozenset([
    "good",
    "suspect",
    "missing",
    "partial",
    "degraded",
    "unknown",
])


class SourceContractError(Exception):
    """Raised when a source object fails contract validation."""

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Contract violation [{field}]: {reason}")


class SourceObjectRecord:
    """
    Immutable record for a single external source data object.

    Validates all required fields and causal constraints at creation time.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = dict(data)
        self._validate()

    def _validate(self) -> None:
        """Validate all required fields and constraints."""
        # Required fields
        for field in REQUIRED_OBJECT_FIELDS:
            if field not in self._data or self._data[field] is None:
                raise SourceContractError(
                    field, f"Required field '{field}' is missing"
                )

        # Validate timestamps
        self._validate_timestamp("valid_time_utc")
        self._validate_timestamp("retrieval_time_utc")
        if self._data.get("issue_time_utc"):
            self._validate_timestamp("issue_time_utc")

        # Validate revision status
        rev = self._data.get("revision_status", "")
        if rev not in VALID_REVISION_STATUSES:
            raise SourceContractError(
                "revision_status",
                f"Invalid revision status '{rev}'. "
                f"Valid: {sorted(VALID_REVISION_STATUSES)}",
            )

        # Validate quality flags
        flags = self._data.get("quality_flags", [])
        if isinstance(flags, str):
            flags = [flags]
            self._data["quality_flags"] = flags
        for flag in flags:
            if flag not in VALID_QUALITY_FLAGS:
                raise SourceContractError(
                    "quality_flags",
                    f"Invalid quality flag '{flag}'. "
                    f"Valid: {sorted(VALID_QUALITY_FLAGS)}",
                )

        # Validate checksum format (64-char hex)
        checksum = self._data.get("response_checksum_sha256", "")
        if len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
            raise SourceContractError(
                "response_checksum_sha256",
                f"Invalid SHA-256 checksum format: '{checksum[:20]}...'",
            )

        # Validate latency is non-negative
        latency = self._data.get("latency_seconds", -1)
        if not isinstance(latency, (int, float)) or latency < 0:
            raise SourceContractError(
                "latency_seconds",
                f"Latency must be non-negative, got {latency}",
            )

    def _validate_timestamp(self, field: str) -> None:
        """Validate an ISO timestamp field."""
        value = self._data.get(field, "")
        if not value:
            raise SourceContractError(field, f"Timestamp '{field}' is empty")
        try:
            self._parse_timestamp(value)
        except (ValueError, TypeError) as e:
            raise SourceContractError(
                field, f"Invalid timestamp format: '{value}' ({e})"
            )

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse an ISO timestamp string to datetime."""
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    @property
    def source_id(self) -> str:
        return self._data["source_id"]

    @property
    def valid_time(self) -> datetime:
        return self._parse_timestamp(self._data["valid_time_utc"])

    @property
    def issue_time(self) -> Optional[datetime]:
        it = self._data.get("issue_time_utc")
        return self._parse_timestamp(it) if it else None

    @property
    def retrieval_time(self) -> datetime:
        return self._parse_timestamp(self._data["retrieval_time_utc"])

    @property
    def checksum(self) -> str:
        return self._data["response_checksum_sha256"]

    @property
    def revision_status(self) -> str:
        return self._data["revision_status"]

    @property
    def quality_flags(self) -> List[str]:
        flags = self._data.get("quality_flags", [])
        return flags if isinstance(flags, list) else [flags]

    @property
    def latency_seconds(self) -> float:
        return float(self._data["latency_seconds"])

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a dictionary."""
        return dict(self._data)

    def is_causally_eligible(
        self, issue_time_utc: datetime, max_ingestion_delay_seconds: float = 0.0
    ) -> bool:
        """
        Check if this source object is causally eligible for a given issue time.

        A source object is eligible only if:
          valid_time <= issue_time
          AND
          retrieval_time <= issue_time + allowed_ingestion_delay
        """
        if self.valid_time > issue_time_utc:
            return False

        allowed_cutoff = issue_time_utc + timedelta(
            seconds=max_ingestion_delay_seconds
        )
        if self.retrieval_time > allowed_cutoff:
            return False

        # Revised data: only original or preliminary is causal
        if self.revision_status == "revised":
            return False

        return True



class SourceManifest:
    """
    Immutable manifest of source objects used in a training or inference run.

    Provides:
      - append-only object tracking
      - deterministic content hash for reproducibility
      - causal eligibility filtering
      - provenance summary
    """

    def __init__(self):
        self._objects: List[SourceObjectRecord] = []
        self._frozen: bool = False

    def add_object(self, data: Dict[str, Any]) -> SourceObjectRecord:
        """Add a source object to the manifest. Validates on add."""
        if self._frozen:
            raise SourceContractError(
                "manifest", "Manifest is frozen. Cannot add objects."
            )
        record = SourceObjectRecord(data)
        self._objects.append(record)
        return record

    def freeze(self) -> str:
        """
        Freeze the manifest. Returns the content hash.

        After freezing, no more objects can be added.
        """
        self._frozen = True
        return self.content_hash

    @property
    def content_hash(self) -> str:
        """Deterministic SHA-256 of all object checksums, sorted."""
        checksums = sorted(obj.checksum for obj in self._objects)
        combined = "|".join(checksums)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @property
    def object_count(self) -> int:
        return len(self._objects)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get_objects_by_source(self, source_id: str) -> List[SourceObjectRecord]:
        """Return all objects for a given source ID."""
        return [obj for obj in self._objects if obj.source_id == source_id]

    def get_causally_eligible_objects(
        self,
        issue_time_utc: datetime,
        max_ingestion_delay_seconds: float = 0.0,
    ) -> List[SourceObjectRecord]:
        """Return only objects that are causally eligible for the given issue time."""
        return [
            obj
            for obj in self._objects
            if obj.is_causally_eligible(issue_time_utc, max_ingestion_delay_seconds)
        ]

    def get_unique_sources(self) -> List[str]:
        """Return unique source IDs in this manifest."""
        return sorted(set(obj.source_id for obj in self._objects))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the manifest to a dictionary."""
        return {
            "manifest_type": "external_source_manifest",
            "frozen": self._frozen,
            "content_hash": self.content_hash,
            "object_count": self.object_count,
            "unique_sources": self.get_unique_sources(),
            "objects": [obj.to_dict() for obj in self._objects],
        }

    def save(self, path: str) -> None:
        """Save the manifest to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "SourceManifest":
        """Load a manifest from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = cls()
        for obj_data in data.get("objects", []):
            manifest.add_object(obj_data)

        if data.get("frozen", False):
            manifest.freeze()

        return manifest

    def verify_against_saved(self, path: str) -> bool:
        """Verify that this manifest matches a previously saved version."""
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        return self.content_hash == saved.get("content_hash", "")
