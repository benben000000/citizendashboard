"""
External Source Registry — Executable License Gate.

Enforces data-rights compliance as a release gate. No external source
may enter production training or inference unless its registry record
has a decision of APPROVED_FREE_COMMERCIAL or
APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION.

The registry rejects:
  - missing records
  - unknown decisions
  - expired reviews
  - contradictory permission fields
  - production use when training or commercial inference is false
  - sources with missing required fields

Every denial returns a structured reason.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
DEFAULT_REGISTRY_PATH = os.path.join(DATA_DIR, "external_source_registry.json")

# ─── Valid decision states ────────────────────────────────────────────
VALID_DECISIONS = frozenset([
    "APPROVED_FREE_COMMERCIAL",
    "APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION",
    "UNKNOWN_BLOCKED",
    "PENDING_REVIEW",
    "RESEARCH_ONLY",
    "NOT_ALLOWED",
])

PRODUCTION_ELIGIBLE_DECISIONS = frozenset([
    "APPROVED_FREE_COMMERCIAL",
    "APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION",
])

BLOCKED_DECISIONS = frozenset([
    "UNKNOWN_BLOCKED",
    "PENDING_REVIEW",
    "RESEARCH_ONLY",
    "NOT_ALLOWED",
])

# ─── Required fields per source record ───────────────────────────────
REQUIRED_SOURCE_FIELDS = [
    "source_id",
    "provider",
    "product",
    "decision",
]

REQUIRED_PERMISSION_FIELDS = [
    "commercial_use_allowed",
    "training_use_allowed",
    "commercial_inference_allowed",
    "derived_features_allowed",
    "raw_caching_allowed",
    "redistribution_allowed",
    "attribution_required",
]

REQUIRED_APPROVED_FIELDS = [
    "source_url",
    "license_url",
    "terms_version_or_date",
    "reviewed_at_utc",
    "review_due_utc",
]


class SourceRegistryError(Exception):
    """Raised when a source fails the registry gate."""

    def __init__(self, source_id: str, reason: str, details: Optional[Dict] = None):
        self.source_id = source_id
        self.reason = reason
        self.details = details or {}
        super().__init__(f"Source '{source_id}' blocked: {reason}")


class ExternalSourceRegistry:
    """
    Executable license gate for external data sources.

    Load the registry JSON and use assert_production_eligible() to
    gate any source before it enters training or inference.
    """

    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH
        self._registry_data: Dict[str, Any] = {}
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._registry_hash: str = ""
        self._load()

    def _load(self) -> None:
        """Load and validate registry JSON."""
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(
                f"External source registry not found: {self.registry_path}"
            )

        with open(self.registry_path, "r", encoding="utf-8") as f:
            raw = f.read()

        self._registry_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        self._registry_data = json.loads(raw)
        self._sources = self._registry_data.get("sources", {})

    @property
    def registry_hash(self) -> str:
        """SHA-256 hash of the registry file for provenance."""
        return self._registry_hash

    @property
    def registry_version(self) -> str:
        """Version string of the registry."""
        return self._registry_data.get("registry_version", "unknown")

    def list_sources(self) -> List[str]:
        """Return all registered source IDs."""
        return list(self._sources.keys())

    def list_eligible_sources(self) -> List[str]:
        """Return source IDs with production-eligible decisions."""
        eligible = []
        for sid, record in self._sources.items():
            decision = record.get("decision", "")
            if decision in PRODUCTION_ELIGIBLE_DECISIONS:
                try:
                    self._validate_approved_record(sid, record)
                    eligible.append(sid)
                except SourceRegistryError:
                    pass
        return eligible

    def list_blocked_sources(self) -> List[str]:
        """Return source IDs that are blocked from production."""
        blocked = []
        for sid, record in self._sources.items():
            decision = record.get("decision", "")
            if decision in BLOCKED_DECISIONS:
                blocked.append(sid)
            elif decision in PRODUCTION_ELIGIBLE_DECISIONS:
                try:
                    self._validate_approved_record(sid, record)
                except SourceRegistryError:
                    blocked.append(sid)
        return blocked

    def get_source(self, source_id: str) -> Dict[str, Any]:
        """Retrieve a source record by ID."""
        if source_id not in self._sources:
            raise SourceRegistryError(
                source_id,
                "Source not found in registry",
                {"available_sources": list(self._sources.keys())},
            )
        return dict(self._sources[source_id])

    def get_decision(self, source_id: str) -> str:
        """Return the decision for a source."""
        record = self.get_source(source_id)
        return record.get("decision", "UNKNOWN_BLOCKED")

    def is_production_eligible(self, source_id: str) -> Tuple[bool, str]:
        """
        Check whether a source is eligible for production use.

        Returns:
            (eligible: bool, reason: str)
        """
        try:
            self.assert_production_eligible(source_id)
            return True, "approved"
        except SourceRegistryError as e:
            return False, e.reason

    def assert_production_eligible(self, source_id: str) -> Dict[str, Any]:
        """
        Assert that a source is eligible for production training and inference.

        Raises SourceRegistryError if the source is blocked for any reason.

        Returns the validated source record on success.
        """
        # 1. Source must exist
        if source_id not in self._sources:
            raise SourceRegistryError(
                source_id,
                "Source not found in registry. Unknown sources are blocked.",
                {"available_sources": list(self._sources.keys())},
            )

        record = self._sources[source_id]

        # 2. Required fields must be present
        for field in REQUIRED_SOURCE_FIELDS:
            if not record.get(field):
                raise SourceRegistryError(
                    source_id,
                    f"Required field '{field}' is missing or empty",
                )

        # 3. Decision must be valid
        decision = record.get("decision", "")
        if decision not in VALID_DECISIONS:
            raise SourceRegistryError(
                source_id,
                f"Unknown decision '{decision}'. Valid decisions: {sorted(VALID_DECISIONS)}",
            )

        # 4. Decision must be production-eligible
        if decision not in PRODUCTION_ELIGIBLE_DECISIONS:
            raise SourceRegistryError(
                source_id,
                f"Decision '{decision}' is not production-eligible. "
                f"Required: {sorted(PRODUCTION_ELIGIBLE_DECISIONS)}",
                {"blocked_reason": record.get("blocked_reason", "")},
            )

        # 5. Validate the approved record in detail
        self._validate_approved_record(source_id, record)

        return record

    def _validate_approved_record(
        self, source_id: str, record: Dict[str, Any]
    ) -> None:
        """Validate an approved source record for consistency and completeness."""
        # 5a. Permission fields must exist
        for field in REQUIRED_PERMISSION_FIELDS:
            if field not in record:
                raise SourceRegistryError(
                    source_id,
                    f"Permission field '{field}' is missing from record",
                )

        # 5b. Training and commercial inference must be allowed
        if not record.get("training_use_allowed"):
            raise SourceRegistryError(
                source_id,
                "Training use is not allowed but decision is production-eligible. "
                "This is a contradictory record.",
            )

        if not record.get("commercial_inference_allowed"):
            raise SourceRegistryError(
                source_id,
                "Commercial inference is not allowed but decision is production-eligible. "
                "This is a contradictory record.",
            )

        if not record.get("commercial_use_allowed"):
            raise SourceRegistryError(
                source_id,
                "Commercial use is not allowed but decision is production-eligible. "
                "This is a contradictory record.",
            )

        # 5c. Approved records must have review metadata
        for field in REQUIRED_APPROVED_FIELDS:
            if not record.get(field):
                raise SourceRegistryError(
                    source_id,
                    f"Approved source is missing required field '{field}'",
                )

        # 5d. Check review expiry
        review_due = record.get("review_due_utc", "")
        if review_due:
            try:
                due_dt = datetime.fromisoformat(review_due.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if now > due_dt:
                    raise SourceRegistryError(
                        source_id,
                        f"License review has expired (due: {review_due}). "
                        "Source must be re-reviewed before production use.",
                        {"review_due_utc": review_due},
                    )
            except ValueError:
                raise SourceRegistryError(
                    source_id,
                    f"Invalid review_due_utc format: '{review_due}'",
                )

    def validate_training_manifest_sources(
        self, source_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that all sources referenced in a training manifest are eligible.

        Returns a summary dict with per-source results.
        """
        results = {
            "all_eligible": True,
            "registry_hash": self._registry_hash,
            "sources": {},
        }

        for sid in source_ids:
            eligible, reason = self.is_production_eligible(sid)
            results["sources"][sid] = {
                "eligible": eligible,
                "reason": reason,
                "decision": self._sources.get(sid, {}).get("decision", "NOT_FOUND"),
            }
            if not eligible:
                results["all_eligible"] = False

        return results

    def revoke_source(self, source_id: str, reason: str = "") -> None:
        """
        Revoke a source's production eligibility in-memory.

        This does NOT write to the file. It is used for runtime policy enforcement
        when a source must be disabled (e.g., latency breach, schema change).
        """
        if source_id not in self._sources:
            raise SourceRegistryError(source_id, "Cannot revoke unknown source")

        self._sources[source_id]["decision"] = "NOT_ALLOWED"
        self._sources[source_id]["blocked_reason"] = (
            f"Runtime revocation: {reason}" if reason else "Runtime revocation"
        )

    def get_provenance_record(self) -> Dict[str, Any]:
        """Return a provenance-safe summary of the registry state."""
        return {
            "registry_hash": self._registry_hash,
            "registry_version": self.registry_version,
            "source_count": len(self._sources),
            "eligible_sources": self.list_eligible_sources(),
            "blocked_sources": self.list_blocked_sources(),
        }
