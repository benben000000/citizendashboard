"""
Causal Feature Cube — Unified Feature Representation with Row Identity.

Enforces causal correctness, source provenance, and deterministic
reproducibility for all feature groups used in training and inference.

Row identity:
    station_id × issue_time_utc × horizon_hours

Every feature group carries:
    - source_id
    - valid_time
    - issue_time
    - retrieval_time
    - availability_cutoff
    - quality_flag
    - missingness_flag
    - license_registry_hash
    - raw_object_hash
    - parser_version

Enforces:
    valid_time <= issue_time
    operationally_available_time <= issue_time + allowed_ingestion_delay
"""

import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List


class CausalViolationError(Exception):
    """Raised when a causal constraint is violated."""

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Causal violation [{field}]: {reason}")


class FeatureGroupMetadata:
    """Metadata that every feature group must carry."""

    def __init__(
        self,
        source_id: str,
        valid_time_utc: datetime,
        issue_time_utc: datetime,
        retrieval_time_utc: datetime,
        availability_cutoff_utc: datetime,
        quality_flag: str = "unknown",
        is_missing: bool = False,
        license_registry_hash: str = "",
        raw_object_hash: str = "",
        parser_version: str = "1.0.0",
    ):
        self.source_id = source_id
        self.valid_time_utc = valid_time_utc
        self.issue_time_utc = issue_time_utc
        self.retrieval_time_utc = retrieval_time_utc
        self.availability_cutoff_utc = availability_cutoff_utc
        self.quality_flag = quality_flag
        self.is_missing = is_missing
        self.license_registry_hash = license_registry_hash
        self.raw_object_hash = raw_object_hash
        self.parser_version = parser_version

    def validate_causality(self, max_ingestion_delay_seconds: float = 0.0) -> None:
        """
        Validate causal constraints.

        Raises CausalViolationError if:
          - valid_time > issue_time
          - retrieval_time > issue_time + allowed_ingestion_delay
        """
        if self.valid_time_utc > self.issue_time_utc:
            raise CausalViolationError(
                "valid_time",
                f"valid_time ({self.valid_time_utc.isoformat()}) is after "
                f"issue_time ({self.issue_time_utc.isoformat()}). "
                "Feature data must be available before or at issue time.",
            )

        allowed_cutoff = self.issue_time_utc + timedelta(
            seconds=max_ingestion_delay_seconds
        )
        if self.retrieval_time_utc > allowed_cutoff:
            raise CausalViolationError(
                "retrieval_time",
                f"retrieval_time ({self.retrieval_time_utc.isoformat()}) exceeds "
                f"allowed ingestion cutoff ({allowed_cutoff.isoformat()}).",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return {
            "source_id": self.source_id,
            "valid_time_utc": self.valid_time_utc.isoformat(),
            "issue_time_utc": self.issue_time_utc.isoformat(),
            "retrieval_time_utc": self.retrieval_time_utc.isoformat(),
            "availability_cutoff_utc": self.availability_cutoff_utc.isoformat(),
            "quality_flag": self.quality_flag,
            "is_missing": self.is_missing,
            "license_registry_hash": self.license_registry_hash,
            "raw_object_hash": self.raw_object_hash,
            "parser_version": self.parser_version,
        }


class FeatureGroup:
    """A group of features with source provenance metadata."""

    def __init__(
        self,
        name: str,
        metadata: FeatureGroupMetadata,
        features: Dict[str, Any],
    ):
        self.name = name
        self.metadata = metadata
        self.features = dict(features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.name,
            "metadata": self.metadata.to_dict(),
            "features": self.features,
        }


class CausalFeatureRow:
    """
    A single row in the causal feature cube.

    Identity: station_id × issue_time_utc × horizon_hours
    """

    def __init__(
        self,
        station_id: str,
        issue_time_utc: datetime,
        horizon_hours: float,
    ):
        self.station_id = station_id
        self.issue_time_utc = issue_time_utc
        self.horizon_hours = horizon_hours
        self._groups: Dict[str, FeatureGroup] = {}
        self._frozen: bool = False

    @property
    def row_key(self) -> str:
        """Deterministic row identifier."""
        return f"{self.station_id}|{self.issue_time_utc.isoformat()}|h{self.horizon_hours}"

    def add_feature_group(
        self,
        group: FeatureGroup,
        max_ingestion_delay_seconds: float = 0.0,
    ) -> None:
        """
        Add a feature group with causal validation.

        Raises CausalViolationError if causal constraints are violated.
        """
        if self._frozen:
            raise CausalViolationError(
                "row", "Feature row is frozen. Cannot add groups."
            )

        # Validate causality
        group.metadata.validate_causality(max_ingestion_delay_seconds)

        self._groups[group.name] = group

    def freeze(self) -> str:
        """Freeze the row and return its content hash."""
        self._frozen = True
        return self.content_hash

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def content_hash(self) -> str:
        """Deterministic SHA-256 hash of all feature groups, sorted by name."""
        parts = []
        for name in sorted(self._groups.keys()):
            group = self._groups[name]
            group_str = json.dumps(group.to_dict(), sort_keys=True, default=str)
            parts.append(group_str)
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get_flat_features(self) -> Dict[str, Any]:
        """
        Return all features flattened into a single dictionary.

        Feature keys are prefixed with the group name.
        """
        flat: Dict[str, Any] = {
            "station_id": self.station_id,
            "issue_time_utc": self.issue_time_utc.isoformat(),
            "horizon_hours": self.horizon_hours,
        }
        for name in sorted(self._groups.keys()):
            group = self._groups[name]
            for key, value in group.features.items():
                flat[f"{name}__{key}"] = value
            flat[f"{name}__source_id"] = group.metadata.source_id
            flat[f"{name}__quality_flag"] = group.metadata.quality_flag
            flat[f"{name}__is_missing"] = group.metadata.is_missing
        return flat

    def get_source_summary(self) -> Dict[str, Any]:
        """Return a summary of sources used in this row."""
        sources = {}
        for name, group in self._groups.items():
            sources[name] = {
                "source_id": group.metadata.source_id,
                "quality_flag": group.metadata.quality_flag,
                "is_missing": group.metadata.is_missing,
                "registry_hash": group.metadata.license_registry_hash,
            }
        return sources

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the row to a dictionary."""
        return {
            "row_key": self.row_key,
            "station_id": self.station_id,
            "issue_time_utc": self.issue_time_utc.isoformat(),
            "horizon_hours": self.horizon_hours,
            "frozen": self._frozen,
            "content_hash": self.content_hash,
            "groups": {
                name: group.to_dict()
                for name, group in sorted(self._groups.items())
            },
        }

    @property
    def group_names(self) -> List[str]:
        return sorted(self._groups.keys())

    @property
    def group_count(self) -> int:
        return len(self._groups)
