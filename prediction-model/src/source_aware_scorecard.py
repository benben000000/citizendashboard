"""
Source-Aware Scorecard — Extension Fields for Multi-Source Evaluation.

Extends the existing validation scorecard with source-aware fields:
  - source_set
  - source_registry_hash
  - source_missing_rate
  - source_latency_p95
  - source_ablation_delta
  - worst_fold_delta
  - confidence_interval
  - sample_count
  - promotion_decision
  - information_status

Promotion rule:
  A target/horizon is promoted only when:
    1. predefined primary metric beats persistence
    2. improvement appears in a majority of rolling folds
    3. worst fold is not materially degraded
    4. source-missing behavior is safe
    5. uncertainty calibration is not materially worse
    6. no leakage detected
    7. provenance and license gates pass

  Otherwise: keep fallback, mark 'Information Limited'.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple


class SourceAwareScorecardEntry:
    """
    A single target/horizon scorecard entry with source-aware fields.
    """

    def __init__(
        self,
        target: str,
        horizon_hours: int,
        source_set: str = "local_only",
        source_registry_hash: str = "",
        source_missing_rate: float = 0.0,
        source_latency_p95: float = 0.0,
        source_ablation_delta: float = 0.0,
        worst_fold_delta: float = 0.0,
        confidence_interval: Tuple[float, float] = (0.0, 0.0),
        sample_count: int = 0,
        persistence_metric: float = 0.0,
        candidate_metric: float = 0.0,
        metric_name: str = "MAE",
        folds_improved: int = 0,
        total_folds: int = 0,
        worst_fold_metric: float = 0.0,
        calibration_delta: float = 0.0,
        leakage_detected: bool = False,
        provenance_pass: bool = False,
        license_pass: bool = False,
    ):
        self.target = target
        self.horizon_hours = horizon_hours
        self.source_set = source_set
        self.source_registry_hash = source_registry_hash
        self.source_missing_rate = source_missing_rate
        self.source_latency_p95 = source_latency_p95
        self.source_ablation_delta = source_ablation_delta
        self.worst_fold_delta = worst_fold_delta
        self.confidence_interval = confidence_interval
        self.sample_count = sample_count
        self.persistence_metric = persistence_metric
        self.candidate_metric = candidate_metric
        self.metric_name = metric_name
        self.folds_improved = folds_improved
        self.total_folds = total_folds
        self.worst_fold_metric = worst_fold_metric
        self.calibration_delta = calibration_delta
        self.leakage_detected = leakage_detected
        self.provenance_pass = provenance_pass
        self.license_pass = license_pass

    def evaluate_promotion(
        self,
        max_degradation_fraction: float = 0.10,
        max_calibration_degradation: float = 0.05,
        max_source_missing_rate: float = 0.30,
    ) -> Tuple[str, str]:
        """
        Evaluate whether this target/horizon should be promoted.

        Returns:
            (decision: str, reason: str)
            decision is one of: 'PROMOTE', 'KEEP_FALLBACK', 'INFORMATION_LIMITED'
        """
        reasons = []

        # 1. Must beat persistence
        # For MAE-type metrics, lower is better
        metric_lower_better = self.metric_name in ("MAE", "RMSE", "median_ae", "Brier", "log_loss")
        if metric_lower_better:
            beats_persistence = self.candidate_metric < self.persistence_metric
        else:
            beats_persistence = self.candidate_metric > self.persistence_metric

        if not beats_persistence:
            reasons.append(
                f"Candidate {self.metric_name}={self.candidate_metric:.4f} does not beat "
                f"persistence={self.persistence_metric:.4f}"
            )

        # 2. Majority of folds improved
        if self.total_folds > 0 and self.folds_improved <= self.total_folds / 2:
            reasons.append(
                f"Only {self.folds_improved}/{self.total_folds} folds improved"
            )

        # 3. Worst fold not materially degraded
        if self.worst_fold_delta > max_degradation_fraction:
            reasons.append(
                f"Worst fold degradation={self.worst_fold_delta:.4f} exceeds "
                f"threshold={max_degradation_fraction:.4f}"
            )

        # 4. Source-missing behavior is safe
        if self.source_missing_rate > max_source_missing_rate:
            reasons.append(
                f"Source missing rate={self.source_missing_rate:.2%} exceeds "
                f"threshold={max_source_missing_rate:.2%}"
            )

        # 5. Calibration not materially worse
        if abs(self.calibration_delta) > max_calibration_degradation:
            reasons.append(
                f"Calibration degradation={self.calibration_delta:.4f} exceeds "
                f"threshold={max_calibration_degradation:.4f}"
            )

        # 6. No leakage
        if self.leakage_detected:
            reasons.append("Leakage detected in candidate model")

        # 7. Provenance and license gates
        if not self.provenance_pass:
            reasons.append("Provenance verification failed")
        if not self.license_pass:
            reasons.append("License gate failed")

        # Decision
        if reasons:
            if self.sample_count < 30:
                return "INFORMATION_LIMITED", (
                    f"Insufficient evidence (n={self.sample_count}). " +
                    "; ".join(reasons)
                )
            return "KEEP_FALLBACK", "; ".join(reasons)
        else:
            return "PROMOTE", "All promotion criteria met"

    @property
    def promotion_decision(self) -> str:
        decision, _ = self.evaluate_promotion()
        return decision

    @property
    def information_status(self) -> str:
        decision, reason = self.evaluate_promotion()
        if decision == "INFORMATION_LIMITED":
            return "Information Limited"
        elif decision == "KEEP_FALLBACK":
            return "Fallback Active"
        else:
            return "Promoted"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary for the scorecard."""
        decision, reason = self.evaluate_promotion()
        return {
            "target": self.target,
            "horizon_hours": self.horizon_hours,
            "source_set": self.source_set,
            "source_registry_hash": self.source_registry_hash,
            "source_missing_rate": self.source_missing_rate,
            "source_latency_p95": self.source_latency_p95,
            "source_ablation_delta": self.source_ablation_delta,
            "worst_fold_delta": self.worst_fold_delta,
            "confidence_interval": list(self.confidence_interval),
            "sample_count": self.sample_count,
            "persistence_metric": self.persistence_metric,
            "candidate_metric": self.candidate_metric,
            "metric_name": self.metric_name,
            "folds_improved": self.folds_improved,
            "total_folds": self.total_folds,
            "worst_fold_metric": self.worst_fold_metric,
            "calibration_delta": self.calibration_delta,
            "leakage_detected": self.leakage_detected,
            "provenance_pass": self.provenance_pass,
            "license_pass": self.license_pass,
            "promotion_decision": decision,
            "information_status": self.information_status,
            "promotion_reason": reason,
        }


class SourceAwareScorecard:
    """
    Collection of source-aware scorecard entries for all target/horizon pairs.
    """

    TARGETS = [
        "temperature", "humidity", "pressure",
        "wind_speed", "wind_direction",
        "heat_index", "rain", "uv", "luminosity",
    ]
    HORIZONS = [1, 3, 6, 12, 24]

    def __init__(self):
        self._entries: Dict[str, SourceAwareScorecardEntry] = {}
        self._generated_at: str = ""

    def add_entry(self, entry: SourceAwareScorecardEntry) -> None:
        """Add or update a scorecard entry."""
        key = f"{entry.target}_h{entry.horizon_hours}"
        self._entries[key] = entry

    def get_entry(
        self, target: str, horizon_hours: int
    ) -> Optional[SourceAwareScorecardEntry]:
        """Retrieve a scorecard entry."""
        key = f"{target}_h{horizon_hours}"
        return self._entries.get(key)

    def get_promotion_summary(self) -> Dict[str, Any]:
        """Get a summary of all promotion decisions."""
        summary: Dict[str, Any] = {
            "promoted": [],
            "fallback": [],
            "information_limited": [],
        }
        for key, entry in sorted(self._entries.items()):
            decision = entry.promotion_decision
            item = {
                "target": entry.target,
                "horizon": entry.horizon_hours,
                "decision": decision,
                "source_set": entry.source_set,
            }
            if decision == "PROMOTE":
                summary["promoted"].append(item)
            elif decision == "INFORMATION_LIMITED":
                summary["information_limited"].append(item)
            else:
                summary["fallback"].append(item)
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire scorecard."""
        return {
            "scorecard_type": "source_aware",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self._entries),
            "entries": {
                key: entry.to_dict()
                for key, entry in sorted(self._entries.items())
            },
            "promotion_summary": self.get_promotion_summary(),
        }

    def save(self, path: str) -> None:
        """Save scorecard to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "SourceAwareScorecard":
        """Load scorecard from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sc = cls()
        for key, entry_data in data.get("entries", {}).items():
            sc.add_entry(SourceAwareScorecardEntry(
                target=entry_data["target"],
                horizon_hours=entry_data["horizon_hours"],
                source_set=entry_data.get("source_set", "local_only"),
                source_registry_hash=entry_data.get("source_registry_hash", ""),
                source_missing_rate=entry_data.get("source_missing_rate", 0.0),
                source_latency_p95=entry_data.get("source_latency_p95", 0.0),
                source_ablation_delta=entry_data.get("source_ablation_delta", 0.0),
                worst_fold_delta=entry_data.get("worst_fold_delta", 0.0),
                confidence_interval=tuple(
                    entry_data.get("confidence_interval", [0.0, 0.0])
                ),
                sample_count=entry_data.get("sample_count", 0),
                persistence_metric=entry_data.get("persistence_metric", 0.0),
                candidate_metric=entry_data.get("candidate_metric", 0.0),
                metric_name=entry_data.get("metric_name", "MAE"),
                folds_improved=entry_data.get("folds_improved", 0),
                total_folds=entry_data.get("total_folds", 0),
                worst_fold_metric=entry_data.get("worst_fold_metric", 0.0),
                calibration_delta=entry_data.get("calibration_delta", 0.0),
                leakage_detected=entry_data.get("leakage_detected", False),
                provenance_pass=entry_data.get("provenance_pass", False),
                license_pass=entry_data.get("license_pass", False),
            ))
        return sc


class TargetRollback:
    """
    Target-specific and horizon-specific rollback capability.

    Tracks the fallback policy for each target/horizon pair and
    can restore it when a candidate fails promotion criteria.
    """

    def __init__(self, policy_path: Optional[str] = None):
        self._fallback_policy: Dict[str, Dict[str, Any]] = {}
        self._active_policy: Dict[str, Dict[str, Any]] = {}
        if policy_path:
            self._load_policy(policy_path)

    def _load_policy(self, path: str) -> None:
        """Load the inference policy from disk."""
        with open(path, "r", encoding="utf-8") as f:
            policy = json.load(f)
        for h_str, h_policy in policy.get("horizons", {}).items():
            for target, source in h_policy.get("selected_sources", {}).items():
                key = f"{target}_h{h_str}"
                self._active_policy[key] = {
                    "target": target,
                    "horizon": int(h_str),
                    "source": source,
                }
                # Fallback is always persistence
                self._fallback_policy[key] = {
                    "target": target,
                    "horizon": int(h_str),
                    "source": "persistence_fallback",
                }

    def get_active_source(self, target: str, horizon: int) -> str:
        """Get the active source for a target/horizon."""
        key = f"{target}_h{horizon}"
        return self._active_policy.get(key, {}).get("source", "persistence_fallback")

    def rollback(self, target: str, horizon: int) -> Dict[str, Any]:
        """
        Rollback a target/horizon to its fallback source.

        Returns the rollback record.
        """
        key = f"{target}_h{horizon}"
        old_source = self._active_policy.get(key, {}).get("source", "unknown")
        fallback = self._fallback_policy.get(key, {})
        self._active_policy[key] = dict(fallback)
        return {
            "target": target,
            "horizon": horizon,
            "old_source": old_source,
            "new_source": fallback.get("source", "persistence_fallback"),
            "action": "rollback",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def rollback_all(self) -> List[Dict[str, Any]]:
        """Rollback all targets to their fallback sources."""
        records = []
        for key in sorted(self._active_policy.keys()):
            parts = key.rsplit("_h", 1)
            if len(parts) == 2:
                target, h_str = parts
                records.append(self.rollback(target, int(h_str)))
        return records

    def verify_rollback(self, target: str, horizon: int) -> bool:
        """Verify that a target/horizon is on its fallback source."""
        key = f"{target}_h{horizon}"
        active = self._active_policy.get(key, {}).get("source", "")
        fallback = self._fallback_policy.get(key, {}).get("source", "persistence_fallback")
        return active == fallback

    def get_policy_summary(self) -> Dict[str, Any]:
        """Return a summary of the current policy state."""
        return {
            "active": {
                k: v["source"]
                for k, v in sorted(self._active_policy.items())
            },
            "fallback": {
                k: v["source"]
                for k, v in sorted(self._fallback_policy.items())
            },
        }
