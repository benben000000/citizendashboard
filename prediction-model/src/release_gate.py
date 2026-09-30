"""
Promotion gate: can a challenger REPLACE the incumbent, channel by channel?

WHY THIS FILE EXISTS
--------------------
This project has had an incumbent/challenger promotion concept whose mechanism
failed *silently*. In ``train_predictive_quality.py`` the Workstream G promotion
audit built ``_target_routes`` by looping over seven 3-field tuples while
unpacking four names, so every run raised
``ValueError: not enough values to unpack`` at the LAST stage -- after every
checkpoint had been written. Each run exited rc=1 and left a complete, valid,
plausible artifact set on disk. The promotion verdict was never computed. A
person reading the run log saw "5 checkpoints saved" and reasonably concluded
the run succeeded.

``test_benchmark_harness.py`` now guards that specific loop's arity. This file
guards the class of failure underneath it: **a decision procedure that can
report success without having produced a decision.**

Five properties follow, and they are the whole design:

  1. ABSENCE IS A VERDICT, NOT A DEFAULT. A metric that is None, absent from the
     mapping, non-finite, or a block with no standard error yields
     ``INSUFFICIENT_EVIDENCE``. There is no code path that turns a missing
     number into a promotion, and no constructor default that supplies one.
  2. FAIL LOUD. ``main()`` returns a non-zero exit code whenever the gate could
     not make a decision (see EXIT CODES). The process never reports success for
     work it did not do.
  3. EVERY CHANNEL IS DECIDED. The grid iterated is (channel x horizon) as
     *supplied by the caller*, not as discovered from the data. A cell the
     candidate cannot evidence gets its own INSUFFICIENT_EVIDENCE verdict and is
     never quietly dropped from the report. A cell the challenger LOSES is
     recorded as REJECT with a reason -- never omitted.
  4. THE TEST SPLIT IS NOT A DECISION INPUT. Every decision input is labelled
     with a split and must be ``validation``. A block labelled ``test`` raises
     ``GateInputError``. Test metrics are carried into the report under
     ``test_reporting_only`` and are structurally unable to reach the rule.
  5. THE BASELINE IS ITSELF GUARDED. A promotion decision is only meaningful
     against a fixed point. ``compare_baseline_to_fresh`` re-scores the
     incumbent and compares it to the recorded record; drift, an absent
     re-scoring, or an unrun check all BLOCK.

THE RULE, EXACTLY AS IMPLEMENTED
--------------------------------
Direction convention, stated once because everything else depends on it:

    mean_difference = mean( incumbent_row_error - candidate_row_error )

POSITIVE means the challenger is better. For an MAE channel the row error is
``|y - yhat|``; for rain occurrence it is the Brier term ``(p - y)^2``, NOT
``|p - y|``. A gate handed the wrong per-row reduction would invert the sign of
the rain decision.

For each (channel, horizon) cell, INDEPENDENTLY:

  Step 0 -- PRECONDITIONS. Any failure yields INSUFFICIENT_EVIDENCE with a
  specific reason; the cell is never allowed to fall through to a comparison:
    * channel or metric name unknown, so the row-error reduction and the
      direction of "better" are not established. Guarding this is deliberate:
      skill-vs-persistence is HIGHER-is-better and MAE is LOWER-is-better, and
      conflating them is how this project once reported a 6x wind win over
      ECMWF that was really 1.4x.
    * candidate or incumbent validation value is None / non-finite
    * the ``full`` paired block is missing or incomplete (n_pairs,
      mean_difference and stderr are all-or-nothing)
    * n_pairs < MIN_PAIRED_ROWS
    * stderr <= 0 or not finite
    * the ``early`` or ``late`` half block is missing or under
      MIN_HALF_PAIRED_ROWS
    * early and late sample ids overlap, are not chronologically ordered, or
      their sizes do not sum to the full period. Overlapping halves would make
      the split check vacuous -- the same "no observable failure" shape as the
      incident.
    * when per-row errors are supplied AND a paired block is supplied, the two
      must agree, to catch rows paired in the wrong order
    * a gate-wide integrity blocker is in force (corpus, policy, baseline drift)
    * the served source for the cell cannot be resolved from the policy

  Step 1 -- DIRECTION. If mean_difference < 0 the challenger is behind over the
  whole validation period: REJECT, reason WORSE_THAN_INCUMBENT. This is checked
  before the magnitude, because it is the actionable description -- pointing a
  reader at "a half regression" when the loss is everywhere would be true and
  useless. The channel keeps serving the incumbent, and the loss is recorded
  rather than omitted: omitting a losing channel is how a partial regression
  gets published.

  Step 2 -- HALF-SPLIT STABILITY. The challenger must be ahead in BOTH
  chronological halves of the validation period:

        mean_difference_early > 0  AND  mean_difference_late > 0

  (a strictly positive mean difference, NOT 2 sigma inside each half -- see
  JUDGEMENT CALLS). Otherwise REJECT, reason HALF_REGRESSION, naming the half
  that failed. A win concentrated in one half of the validation period is not a
  win.

  Step 3 -- SIGNIFICANCE. PROMOTE requires

        mean_difference > SIGMA_MULTIPLIER * stderr

  STRICTLY greater. A cell sitting exactly at 2 sigma does NOT promote. At
  exactly 2 sigma the one-sided p is 0.0228, and this gate is built so that the
  cost of a wrong promotion exceeds the cost of a missed one. Otherwise:
  REJECT, reason NOT_SIGNIFICANT_AT_2_SIGMA.

Between Steps 2 and 3 the gate also re-derives the paired statistic from any
supplied per-row errors and refuses the cell if it disagrees with the block that
carries it -- a mis-paired column produces a confident, well-formatted, wrong
verdict, which is worse than an absence.

Only Steps 2 and 3 together produce PROMOTE, reason PROMOTED_2_SIGMA_BOTH_HALVES.

OVERALL RECOMMENDATION
----------------------
  PROMOTE_ALL   every decided cell promoted, none rejected
  SPLIT         at least one promoted AND at least one rejected. This is the
                expected shape of a good candidate -- better at wind, worse at
                rain -- and a wholesale rejection would throw the wind gain away
                while keeping the rain regression.
  RETAIN_ALL    no cell promoted; the incumbent keeps every cell
  BLOCKED       at least one cell is INSUFFICIENT_EVIDENCE, or an integrity
                check did not pass. A blocked run has NOT evaluated promotion.

EXIT CODES
----------
  0  gate ran, decision complete. Caller may act on the listed promotions.
  1  BLOCKED. The gate could not make a decision (missing evidence, corpus
     mismatch, policy mismatch, or a failed/absent baseline re-scoring).
  2  RETAIN_ALL. The gate decided, and the challenger lost everywhere.
  3  UNUSABLE_INPUT. The gate could not run: malformed inputs, or a decision
     input labelled with a split other than ``validation``.

Only 0 means "the gate ran and permitted the promotion it listed". A naive
``if rc == 0: deploy`` is therefore safe.

JUDGEMENT CALLS (places the written rule was ambiguous)
-------------------------------------------------------
  A. "beats it by a paired 2-sigma standard error AND does so in BOTH
     chronological halves". Read as: 2-sigma on the FULL validation period, plus
     a strictly-positive mean difference in each half. Requiring 2-sigma inside
     each half too would need roughly 4x the rows for the same power, on
     autocorrelated series whose halves are not independent, and in practice
     would refuse to promote anything at all. The asymmetry is deliberate: a
     gate that never fires is not obviously safer than one that fires rarely,
     so the strictness is concentrated on Step 1 and Step 2 only checks the
     SIGN, not the magnitude.
  B. "per-channel per-horizon test metrics" is the reporting input; the rule
     forbids deciding on test data. So each model carries BOTH a validation and
     a test metric per cell. Only the validation metric and the validation
     paired blocks are readable by the rule.
  C. Rain occurrence is scored by Brier, and its per-row error is the squared
     term. Handing the gate ``|p - y|`` for rain would make the Brier decision
     meaningless while still producing a confident verdict, so the reduction is
     declared per channel and checked against the channel name.
  D. The channel grid is an explicit caller-supplied input, not inferred from
     the data. Inferring it means a channel whose metrics failed to load simply
     does not appear, which is the exact omission this gate exists to prevent.
  E. Served-source selection is recorded and classified but does NOT change the
     verdict. 13 of 20 cells in the reference baseline are served by
     ``persistence_fallback``; for those the challenger is beating persistence,
     not beating a model, and release_baseline.md is explicit that those are two
     different questions. The gate refuses to let the reader lose that
     distinction -- every promotion is tagged ``BEATS_PERSISTENCE`` or
     ``BEATS_SERVED_MODEL`` -- but suppressing a legitimate persistence win
     would be a second, unstated promotion policy.
  F. A cell the incumbent scores but the fresh re-scoring omits is reported as
     ``missing`` and BLOCKS. A cell the fresh re-scoring covers that the
     recorded record does not mention is reported as ``newly_covered`` and does
     NOT block: the recorded document was never expected to describe it, so its
     appearance is not evidence that anything moved underneath.

VOLATILE BASELINE FIELDS
------------------------
``capture_release_baseline.py --check`` excludes exactly one line, the
``- captured:`` timestamp, because an earlier version reported drift on every
single run and an alarm that always fires is not an alarm. The equivalent trap
exists here at the record level: ``compare_baseline_documents`` skips the fields
in VOLATILE_BASELINE_FIELDS and compares everything else. A re-capture must not
read as a regression, and a changed corpus sha or a changed served number must.

READ-ONLY apart from the single JSON verdict file it is asked to write. It never
trains, never fetches, and never mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
DEFAULT_BASELINE_PATH = os.path.join(DATA_DIR, "release_baseline.md")
DEFAULT_POLICY_PATH = os.path.join(DATA_DIR, "inference_policy.json")

# --------------------------------------------------------------------------- #
# The rule's constants. Named, documented, and echoed verbatim into the verdict
# JSON so the published document can never describe a different rule from the
# one that ran.
# --------------------------------------------------------------------------- #

HORIZONS: Tuple[int, ...] = (1, 3, 6, 12, 24)
SIGMA_MULTIPLIER: float = 2.0
MIN_PAIRED_ROWS: int = 30
MIN_HALF_PAIRED_ROWS: int = 10
DRIFT_REL_TOLERANCE: float = 0.01
HALF_KEYS: Tuple[str, str] = ("early", "late")
SPLIT_DECISION: str = "validation"
SPLIT_REPORTING: str = "test"

# Verdict vocabulary. INSUFFICIENT_EVIDENCE is a first-class verdict, not an
# error string: it is the answer when the evidence does not exist.
V_PROMOTE = "PROMOTE"
V_REJECT = "REJECT"
V_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

REASON_PROMOTED = "PROMOTED_2_SIGMA_BOTH_HALVES"
REASON_NOT_SIGNIFICANT = "NOT_SIGNIFICANT_AT_2_SIGMA"
REASON_WORSE = "WORSE_THAN_INCUMBENT"
REASON_HALF_REGRESSION = "HALF_REGRESSION"

R_PROMOTE_ALL = "PROMOTE_ALL"
R_SPLIT = "SPLIT"
R_RETAIN_ALL = "RETAIN_ALL"
R_BLOCKED = "BLOCKED"

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_RETAIN = 2
EXIT_UNUSABLE = 3

EXIT_MEANING = {
    EXIT_OK: "gate ran; decision complete; act on the listed promotions",
    EXIT_BLOCKED: "BLOCKED: the gate could not make a decision",
    EXIT_RETAIN: "the gate decided and the challenger lost every cell",
    EXIT_UNUSABLE: "UNUSABLE_INPUT: the gate could not run",
}

# Metadata that legitimately changes on every capture and must never be part of
# a drift comparison. Mirrors the `- captured:` exclusion in
# capture_release_baseline.py --check.
VOLATILE_BASELINE_FIELDS: Tuple[str, ...] = (
    "captured",
    "captured_at",
    "captured_at_utc",
    "recorded_at_utc",
    "generated_at",
    "generated_at_utc",
    "evaluated_at_utc",
    "timestamp",
    "timestamp_utc",
)

DRIFT_MATCH = "MATCH"
DRIFT_DRIFT = "DRIFT"
DRIFT_NOT_CHECKED = "NOT_CHECKED"
DRIFT_SKIPPED = "SKIPPED_WITH_JUSTIFICATION"

BEATS_PERSISTENCE = "BEATS_PERSISTENCE"
BEATS_SERVED_MODEL = "BEATS_SERVED_MODEL"


class GateInputError(ValueError):
    """The gate could not run at all. Always exit EXIT_UNUSABLE.

    Raised only for defects that make the question unaskable -- a decision input
    labelled with the wrong split, a value that is not a real number, a
    malformed policy. Missing *evidence* is NOT this: a missing metric is a
    verdict, and gets one.
    """


class ChannelSpec:
    """How a channel is scored, and what one of its rows contributes.

    ``row_error`` is the per-row reduction whose mean is the reported metric.
    Getting it wrong is the units class of bug this project already shipped once
    (wind quoted in km/h against m/s telemetry), so it is declared here rather
    than assumed at the call site.
    """

    __slots__ = ("metric", "unit", "better", "row_error")

    def __init__(self, metric: str, unit: str, better: str, row_error: str) -> None:
        self.metric = metric
        self.unit = unit
        self.better = better
        self.row_error = row_error

    def to_dict(self) -> Dict[str, str]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "better": self.better,
            "row_error": self.row_error,
        }


CHANNEL_SPECS: Dict[str, ChannelSpec] = {
    "temperature": ChannelSpec("mae", "degC", "lower_is_better", "absolute"),
    "humidity": ChannelSpec("mae", "%", "lower_is_better", "absolute"),
    "pressure": ChannelSpec("mae", "hPa", "lower_is_better", "absolute"),
    "wind_speed": ChannelSpec("mae", "m/s", "lower_is_better", "absolute"),
    "rain_occurrence": ChannelSpec(
        "brier_score", "dimensionless", "lower_is_better", "squared"
    ),
}

DEFAULT_CHANNELS: Tuple[str, ...] = tuple(sorted(CHANNEL_SPECS))

LEARNED_SOURCES = {"learned_model", "candidate", "refit_policy"}
PERSISTENCE_SOURCES = {"persistence_fallback", "persistence", "persist"}


# --------------------------------------------------------------------------- #
# Rule description, published with every verdict
# --------------------------------------------------------------------------- #

RULE: Dict[str, Any] = {
    "statement": (
        "A challenger replaces the incumbent for a (channel, horizon) cell only "
        "if it beats it by more than a paired 2-sigma standard error on the "
        "validation period AND is ahead in both chronological halves of that "
        "period. Every channel is decided independently."
    ),
    "difference": (
        "mean_difference = mean(incumbent_row_error - candidate_row_error); "
        "positive means the challenger is better"
    ),
    "row_error": (
        "absolute |y - yhat| for MAE channels; Brier term (p - y)^2 for "
        "rain_occurrence"
    ),
    "sigma_multiplier": SIGMA_MULTIPLIER,
    "significance_comparison": (
        "strictly greater than; a cell sitting exactly at the multiple does NOT "
        "promote"
    ),
    "half_split_requirement": (
        "mean_difference > 0 in the 'early' and 'late' halves independently; "
        "the magnitude of each half is not itself tested at 2 sigma"
    ),
    "min_paired_rows": MIN_PAIRED_ROWS,
    "min_paired_rows_per_half": MIN_HALF_PAIRED_ROWS,
    "decision_split": SPLIT_DECISION,
    "test_split": "reporting only; a decision input labelled 'test' is rejected",
    "missing_evidence": (
        "INSUFFICIENT_EVIDENCE; there is no default that turns absence into a "
        "promotion"
    ),
    "rejected_cells": "recorded explicitly with a reason; never omitted",
}


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #


def _optional_number(value: Any, label: str) -> Optional[float]:
    """None, or a finite float. bool is not a number; NaN/inf are not numbers."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise GateInputError("%s is a bool, not a metric" % label)
    if not isinstance(value, (int, float)):
        raise GateInputError(
            "%s is %s, not a number" % (label, type(value).__name__)
        )
    out = float(value)
    if not math.isfinite(out):
        raise GateInputError("%s is %s, which is not a finite metric" % (label, value))
    return out


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateInputError("%s must be a non-empty string, got %r" % (label, value))
    return value


@dataclass(frozen=True)
class CellMetric:
    """One model's scalar score for one (channel, horizon) cell.

    ``value`` is the point estimate. ``None`` is a legitimate and load-bearing
    state: it means the evidence does not exist, and it can only ever produce
    INSUFFICIENT_EVIDENCE.
    """

    value: Optional[float] = None
    n_rows: Optional[int] = None
    split: str = SPLIT_DECISION
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value", _optional_number(self.value, "CellMetric.value")
        )
        if self.split not in (SPLIT_DECISION, SPLIT_REPORTING):
            raise GateInputError(
                "CellMetric.split is %r; expected %r (decisions) or %r (reporting)"
                % (self.split, SPLIT_DECISION, SPLIT_REPORTING)
            )
        if self.n_rows is not None and (
            isinstance(self.n_rows, bool) or not isinstance(self.n_rows, int)
        ):
            raise GateInputError("CellMetric.n_rows must be an int or None")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "n_rows": self.n_rows,
            "split": self.split,
            "source": self.source,
        }


@dataclass(frozen=True)
class PairedBlock:
    """Paired statistics for one cell over one span of the validation period.

    ``n_pairs``, ``mean_difference`` and ``stderr`` are all-or-nothing. A block
    that carries a mean but no standard error is not a weak result, it is an
    unanswerable one, and the gate says so instead of inventing a threshold.
    """

    n_pairs: Optional[int] = None
    mean_difference: Optional[float] = None
    stderr: Optional[float] = None
    split: str = SPLIT_DECISION
    source: str = ""
    sample_ids: Optional[Tuple[str, ...]] = None
    incumbent_errors: Optional[Tuple[float, ...]] = None
    candidate_errors: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        present = [self.n_pairs, self.mean_difference, self.stderr]
        if any(p is None for p in present) and not all(p is None for p in present):
            raise GateInputError(
                "PairedBlock is partial (n_pairs=%r, mean_difference=%r, "
                "stderr=%r); supply all three or none"
                % (self.n_pairs, self.mean_difference, self.stderr)
            )
        if self.n_pairs is not None:
            if isinstance(self.n_pairs, bool) or not isinstance(self.n_pairs, int):
                raise GateInputError("PairedBlock.n_pairs must be an int or None")
            if self.n_pairs < 0:
                raise GateInputError("PairedBlock.n_pairs is negative")
        object.__setattr__(
            self,
            "mean_difference",
            _optional_number(self.mean_difference, "PairedBlock.mean_difference"),
        )
        object.__setattr__(
            self, "stderr", _optional_number(self.stderr, "PairedBlock.stderr")
        )
        if self.split != SPLIT_DECISION:
            raise GateInputError(
                "PairedBlock.split is %r. The promotion decision is made on the "
                "%r split; the test split is for reporting only. A block labelled "
                "%r is a decision input on the wrong data and the gate will not "
                "run it." % (self.split, SPLIT_DECISION, self.split)
            )

    @property
    def complete(self) -> bool:
        return (
            self.n_pairs is not None
            and self.mean_difference is not None
            and self.stderr is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "mean_difference": self.mean_difference,
            "stderr": self.stderr,
            "split": self.split,
            "source": self.source,
            "has_sample_ids": self.sample_ids is not None,
            "has_per_row_errors": self.incumbent_errors is not None,
        }


@dataclass(frozen=True)
class CellEvidence:
    """Everything one model supplies about one (channel, horizon) cell."""

    validation: CellMetric = field(default_factory=CellMetric)
    test: Optional[CellMetric] = None
    paired: Mapping[str, PairedBlock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.test is not None and self.test.split != SPLIT_REPORTING:
            raise GateInputError(
                "CellEvidence.test must be a test-split metric, got split=%r"
                % self.test.split
            )
        if self.validation.split != SPLIT_DECISION:
            raise GateInputError(
                "CellEvidence.validation must be a %r-split metric, got %r"
                % (SPLIT_DECISION, self.validation.split)
            )
        for key, block in dict(self.paired).items():
            if not isinstance(block, PairedBlock):
                raise GateInputError("CellEvidence.paired[%r] is not a PairedBlock" % key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation": self.validation.to_dict(),
            "test": None if self.test is None else self.test.to_dict(),
            "paired": {k: v.to_dict() for k, v in sorted(dict(self.paired).items())},
        }


# --------------------------------------------------------------------------- #
# Served source, from inference_policy.json
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServedSource:
    """What the incumbent actually serves for one cell."""

    source: str
    detail: str
    origin: str  # "selected_sources" | "rain_blend_weights" | ...

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "detail": self.detail,
            "origin": self.origin,
            "is_persistence": self.source in PERSISTENCE_SOURCES,
        }


@dataclass(frozen=True)
class PolicyView:
    """The served-source selection, indexed by (channel, horizon).

    Only channels in CHANNEL_SPECS are indexed. Everything the policy names
    that the gate cannot decide -- wind_direction (circular MAE is structurally
    uninformative against persistence), heat_index and light_intensity (derived
    or gated) -- is listed in ``undecidable_channels`` rather than dropped in
    silence, so the omission is visible in the published verdict.
    """

    policy_version: str = ""
    weather_telemetry_sha256: Optional[str] = None
    sources: Mapping[Tuple[str, int], ServedSource] = field(default_factory=dict)
    read_error: Optional[str] = None
    undecidable_channels: Tuple[str, ...] = ()

    def resolve(self, channel: str, horizon: int) -> Optional[ServedSource]:
        return dict(self.sources).get((channel, horizon))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "weather_telemetry_sha256": self.weather_telemetry_sha256,
            "cells_resolved": len(self.sources),
            "read_error": self.read_error,
            "undecidable_channels": list(self.undecidable_channels),
        }


def load_policy(path: str = DEFAULT_POLICY_PATH) -> PolicyView:
    """Read inference_policy.json into a (channel, horizon) served-source map.

    A missing or malformed file is reported as an empty view carrying a
    ``read_error``. It does not raise, because a gate that dies on a bad policy
    file gives the operator an exception rather than a verdict -- and the
    verdict is the thing that has to survive. Every cell then resolves to
    nothing and is reported INSUFFICIENT_EVIDENCE, which blocks.
    """
    if not os.path.exists(path):
        return PolicyView(read_error="policy file not found: %s" % path)
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        return PolicyView(read_error="could not read %s: %s" % (path, exc))
    if not isinstance(doc, dict):
        return PolicyView(read_error="%s is not a JSON object" % path)

    version = doc.get("policy_version")
    corpus = (doc.get("dataset_hashes") or {}).get("weather_telemetry_sha256")

    sources: Dict[Tuple[str, int], ServedSource] = {}
    undecidable: set = set()
    for h_key, h_block in (doc.get("horizons") or {}).items():
        try:
            horizon = int(h_key)
        except (TypeError, ValueError):
            continue
        selected = (h_block or {}).get("selected_sources") or {}
        for channel, source in selected.items():
            if not isinstance(source, str):
                continue
            # Derived and blocked targets are not independently served: they are
            # computed from the channels above them. Deciding them here would
            # duplicate a verdict under a second name.
            if source.startswith("derived_") or source in ("blocked", "daylight_beta"):
                undecidable.add(channel)
                continue
            if channel not in CHANNEL_SPECS:
                # wind_direction is served and scored, but the gate holds no
                # rule for it; indexing it would invite a verdict nobody defined.
                undecidable.add(channel)
                continue
            sources[(channel, horizon)] = ServedSource(
                source=source, detail=source, origin="selected_sources"
            )
        # Rain is not a selected_source: it is a blend of the rain model and
        # persistence with weights. The served behaviour IS real, so the gate
        # records it rather than pretending the channel is unspecified.
        m_w = (h_block or {}).get("rain_model_weight")
        p_w = (h_block or {}).get("rain_persistence_weight")
        if isinstance(m_w, (int, float)) and isinstance(p_w, (int, float)):
            sources[("rain_occurrence", horizon)] = ServedSource(
                source="rain_blend",
                detail="rain_model_weight=%s, rain_persistence_weight=%s" % (m_w, p_w),
                origin="rain_blend_weights",
            )

    return PolicyView(
        policy_version=version if isinstance(version, str) else "",
        weather_telemetry_sha256=corpus if isinstance(corpus, str) else None,
        sources=sources,
        undecidable_channels=tuple(sorted(undecidable)),
    )


# --------------------------------------------------------------------------- #
# The incumbent record: data/release_baseline.md
# --------------------------------------------------------------------------- #

_SHA_RE = re.compile(r"^- corpus sha256:\s*`([0-9a-f]{64})`", re.MULTILINE)
_CAPTURED_RE = re.compile(r"^- captured:\s*(\S+)\s*$", re.MULTILINE)
_COMMIT_RE = re.compile(r"^- commit:\s*`([^`]*)`", re.MULTILINE)
_CKPT_RE = re.compile(r"h(\d+)=`([0-9a-f]+)`")
# | +6h | 1.477 | 1.699 | +13.0% | 1.274 | 1.696 | 6/9 |
_MAE_ROW_RE = re.compile(
    r"^\|\s*\+(\d+)h\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([+-][0-9.]+)%\s*\|",
    re.MULTILINE,
)
# | +1h | 0.1373 | 0.1633 | 0.2453 | 0.2416 |
_BRIER_ROW_RE = re.compile(
    r"^\|\s*\+(\d+)h\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|",
    re.MULTILINE,
)

_MD_CHANNELS = {
    "temperature": "MAE degC",
    "humidity": "MAE %",
    "pressure": "MAE hPa",
    "wind_speed": "MAE m/s",
}


@dataclass(frozen=True)
class BaselineCell:
    """One row of the incumbent scoreboard."""

    channel: str
    horizon: int
    model: Optional[float]
    persistence: Optional[float]
    skill_pct: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "horizon": int(self.horizon),
            "model": self.model,
            "persistence": self.persistence,
            "skill_pct": self.skill_pct,
        }


@dataclass(frozen=True)
class BaselineDocument:
    """data/release_baseline.md, parsed.

    ``captured`` is parsed and published but is never part of a comparison --
    see VOLATILE_BASELINE_FIELDS and the `--check` note above.
    """

    path: str = ""
    corpus_sha256: Optional[str] = None
    captured: Optional[str] = None
    commit: Optional[str] = None
    checkpoint_sha256: Mapping[int, str] = field(default_factory=dict)
    cells: Mapping[Tuple[str, int], BaselineCell] = field(default_factory=dict)
    parse_errors: Tuple[str, ...] = ()

    def value(self, channel: str, horizon: int) -> Optional[float]:
        cell = dict(self.cells).get((channel, horizon))
        return None if cell is None else cell.model

    def metadata(self) -> Dict[str, Any]:
        return {
            "corpus_sha256": self.corpus_sha256,
            "captured": self.captured,
            "commit": self.commit,
            "checkpoint_sha256": {
                str(h): s for h, s in sorted(dict(self.checkpoint_sha256).items())
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.metadata())
        out["path"] = self.path
        out["cells"] = len(self.cells)
        out["parse_errors"] = list(self.parse_errors)
        return out


def _strip_volatile(doc: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in dict(doc).items()
        if k not in VOLATILE_BASELINE_FIELDS and k != "path"
    }


def load_release_baseline(path: str = DEFAULT_BASELINE_PATH) -> BaselineDocument:
    """Parse the incumbent scoreboard.

    A section that is present but yields no rows is a parse error, not an empty
    result. The original capture script skips those rows with a bare
    `continue`, which means a corrupted table would read as "the incumbent has no
    humidity data" instead of "this file is damaged" -- and the first reading
    looks like a policy change.
    """
    if not os.path.exists(path):
        return BaselineDocument(path=path, parse_errors=("baseline not found: %s" % path,))
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    sha = _SHA_RE.search(text)
    captured = _CAPTURED_RE.search(text)
    commit = _COMMIT_RE.search(text)
    ckpt_line = re.search(r"^- served bundles checkpoint sha256[^\n]*$", text, re.MULTILINE)
    checkpoints: Dict[int, str] = {}
    if ckpt_line:
        for h, digest in _CKPT_RE.findall(ckpt_line.group(0)):
            checkpoints[int(h)] = digest

    cells: Dict[Tuple[str, int], BaselineCell] = {}
    errors: List[str] = []

    for channel, heading in _MD_CHANNELS.items():
        marker = "### %s (%s)" % (channel, heading)
        if marker not in text:
            errors.append("baseline is missing the %r section" % marker)
            continue
        tail = text.split(marker, 1)[1]
        block = tail.split("\n### ", 1)[0]
        rows = _MAE_ROW_RE.findall(block)
        if not rows:
            errors.append("baseline section %r produced no parsable rows" % marker)
        for horizon, model, persist, skill in rows:
            cells[(channel, int(horizon))] = BaselineCell(
                channel=channel,
                horizon=int(horizon),
                model=float(model),
                persistence=float(persist),
                skill_pct=float(skill),
            )

    rain_marker = "### rain occurrence (Brier score, lower is better)"
    if rain_marker in text:
        block = text.split(rain_marker, 1)[1].split("\n## ", 1)[0]
        rows = _BRIER_ROW_RE.findall(block)
        if not rows:
            errors.append("baseline rain section produced no parsable rows")
        for horizon, model, persist, clim, nwp in rows:
            cells[("rain_occurrence", int(horizon))] = BaselineCell(
                channel="rain_occurrence",
                horizon=int(horizon),
                model=float(model),
                persistence=float(persist),
                skill_pct=None,
            )
    else:
        errors.append("baseline is missing the rain occurrence section")

    if not cells:
        errors.append("baseline yielded no cells at all")

    return BaselineDocument(
        path=path,
        corpus_sha256=sha.group(1) if sha else None,
        captured=captured.group(1) if captured else None,
        commit=commit.group(1) if commit else None,
        checkpoint_sha256=checkpoints,
        cells=cells,
        parse_errors=tuple(errors),
    )


# --------------------------------------------------------------------------- #
# Regression guard: is the incumbent record still true?
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DriftReport:
    """Result of re-scoring the incumbent and comparing it to the record."""

    status: str
    tolerance_rel: float = DRIFT_REL_TOLERANCE
    checked: int = 0
    drifted: Tuple[Tuple[str, int], ...] = ()
    missing: Tuple[Tuple[str, int], ...] = ()
    newly_covered: Tuple[Tuple[str, int], ...] = ()
    details: Mapping[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)
    justification: Optional[str] = None
    excluded_fields: Tuple[str, ...] = VOLATILE_BASELINE_FIELDS

    @property
    def clean(self) -> bool:
        return self.status == DRIFT_MATCH

    def to_dict(self) -> Dict[str, Any]:
        key = lambda k: "%s|h%d" % k  # noqa: E731
        return {
            "status": self.status,
            "tolerance_rel": self.tolerance_rel,
            "cells_checked": self.checked,
            "cells_total": (
                self.checked
                + len(self.drifted)
                + len(self.missing)
                + len(self.newly_covered)
            ),
            "drifted": [key(k) for k in self.drifted],
            "not_rechecked": [key(k) for k in self.missing],
            "newly_covered": [key(k) for k in self.newly_covered],
            "justification": self.justification,
            "excluded_fields": list(self.excluded_fields),
            "details": {key(k): v for k, v in sorted(dict(self.details).items())},
        }


def compare_baseline_to_fresh(
    recorded: BaselineDocument,
    fresh: Optional[Mapping[Tuple[str, int], Optional[float]]],
    tolerance_rel: float = DRIFT_REL_TOLERANCE,
    justification: Optional[str] = None,
) -> DriftReport:
    """Re-score guard.

    ``fresh`` maps (channel, horizon) to a freshly computed value, or None where
    the fresh scoring produced nothing. Callers that cannot re-score the
    incumbent may pass ``justification``; the report is then
    SKIPPED_WITH_JUSTIFICATION and BLOCKS anyway. There is no spelling of
    "I did not run the guard" that reads as a pass, which is the point: this
    guard exists to catch a corpus or scoring change that moved underneath an
    unchanged model, and a guard that can be skipped silently cannot.
    """
    if fresh is None:
        return DriftReport(
            status=DRIFT_SKIPPED,
            tolerance_rel=tolerance_rel,
            justification=justification,
        )
    if not dict(fresh):
        return DriftReport(
            status=DRIFT_NOT_CHECKED,
            tolerance_rel=tolerance_rel,
            justification=justification,
        )

    drifted: List[Tuple[str, int]] = []
    missing: List[Tuple[str, int]] = []
    newly: List[Tuple[str, int]] = []
    details: Dict[Tuple[str, int], Dict[str, Any]] = {}
    checked = 0

    for key, cell in dict(recorded.cells).items():
        want = cell.model
        if want is None:
            continue
        if key not in dict(fresh):
            # Recorded but never re-scored: the guard did not cover this cell,
            # so the record is unverified for it. Blocks.
            missing.append(key)
            details[key] = {"status": "NOT_RECHECKED", "recorded": want}
            continue
        got = dict(fresh)[key]
        if got is None:
            missing.append(key)
            details[key] = {"status": "FRESH_VALUE_MISSING", "recorded": want}
            continue
        got = _optional_number(got, "fresh[%s|h%d]" % key)
        assert got is not None
        allowed = tolerance_rel * max(abs(want), 1e-12)
        delta = got - want
        ok = abs(delta) <= allowed
        checked += 1
        details[key] = {
            "status": DRIFT_MATCH if ok else DRIFT_DRIFT,
            "recorded": want,
            "fresh": got,
            "delta": delta,
            "allowed": allowed,
        }
        if not ok:
            drifted.append(key)

    for key in dict(fresh):
        if key in dict(recorded.cells):
            continue
        got = dict(fresh)[key]
        if got is None:
            continue
        newly.append(key)
        details[key] = {"status": "NEWLY_COVERED", "fresh": got}

    if drifted or missing:
        status = DRIFT_DRIFT
    else:
        status = DRIFT_MATCH
    return DriftReport(
        status=status,
        tolerance_rel=tolerance_rel,
        checked=checked,
        drifted=tuple(drifted),
        missing=tuple(missing),
        newly_covered=tuple(newly),
        details=details,
        justification=justification,
    )


def compare_baseline_documents(
    recorded: BaselineDocument, fresh: BaselineDocument
) -> DriftReport:
    """Record-level drift: re-capture the baseline and diff the two documents.

    Only the volatile fields in VOLATILE_BASELINE_FIELDS are excluded. A
    differing corpus sha256 or checkpoint hash is drift and is reported as such;
    a differing capture timestamp is not, because it changes on every run and an
    alarm that always fires is not an alarm.
    """
    a, b = _strip_volatile(recorded.metadata()), _strip_volatile(fresh.metadata())
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    scoreboard_diff = sorted(
        "%s|h%d" % k
        for k in set(dict(recorded.cells)) | set(dict(fresh.cells))
        if dict(recorded.cells).get(k) != dict(fresh.cells).get(k)
    )
    drifted = tuple(
        tuple(int(p) if p.isdigit() else p for p in item.split("|"))  # type: ignore[misc]
        for item in scoreboard_diff
    )
    if differing or scoreboard_diff:
        return DriftReport(
            status=DRIFT_DRIFT,
            checked=len(recorded.cells),
            drifted=drifted,
            details={
                (k, 0): {"status": DRIFT_DRIFT, "field": k,
                         "recorded": a.get(k), "fresh": b.get(k)}
                for k in differing
            },
        )
    return DriftReport(status=DRIFT_MATCH, checked=len(recorded.cells))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateInputs:
    """Everything the rule reads.

    ``drift`` is required and has no default. A caller that cannot re-score the
    incumbent must construct an explicit SKIPPED_WITH_JUSTIFICATION report, which
    is published in the verdict and blocks promotion.
    """

    corpus_sha256: str
    candidate: Mapping[Tuple[str, int], CellEvidence]
    incumbent: Mapping[Tuple[str, int], CellEvidence]
    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    horizons: Tuple[int, ...] = HORIZONS
    baseline: Optional[BaselineDocument] = None
    policy: Optional[PolicyView] = None
    drift: Optional[DriftReport] = None
    candidate_name: str = "challenger"
    incumbent_name: str = "incumbent"

    def cells(self) -> List[Tuple[str, int]]:
        """The grid to decide. Supplied by the caller, never inferred.

        Deriving the grid from whichever keys happen to be populated is how a
        channel with unevaluable metrics disappears from the report instead of
        appearing in it as INSUFFICIENT_EVIDENCE.
        """
        out: List[Tuple[str, int]] = []
        for channel in self.channels:
            for horizon in self.horizons:
                out.append((channel, horizon))
        return out

    def __post_init__(self) -> None:
        # An empty or whitespace corpus SHA is allowed through here so the gate
        # can REPORT it as CORPUS_SHA_MISSING in the verdict document, naming
        # the omission. A non-string is a malformed input and is refused.
        if not isinstance(self.corpus_sha256, str):
            raise GateInputError(
                "GateInputs.corpus_sha256 must be a string, got %s"
                % type(self.corpus_sha256).__name__
            )
        if not self.channels:
            raise GateInputError(
                "GateInputs.channels is empty; a gate with nothing to decide is "
                "not a decision, and returning one would be the failure this "
                "module exists to prevent"
            )
        if not self.horizons:
            raise GateInputError(
                "GateInputs.horizons is empty; see the note on channels"
            )
        for channel in self.channels:
            if not isinstance(channel, str) or not channel.strip():
                raise GateInputError("GateInputs.channels contains %r" % (channel,))
        for horizon in self.horizons:
            if isinstance(horizon, bool) or not isinstance(horizon, int):
                raise GateInputError("GateInputs.horizons contains %r" % (horizon,))


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellVerdict:
    channel: str
    horizon: int
    verdict: str
    reason: str
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, int]:
        return (self.channel, self.horizon)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "channel": self.channel,
            "horizon": int(self.horizon),
            "verdict": self.verdict,
            "reason": self.reason,
        }
        if self.detail:
            out["detail"] = self.detail
        if self.evidence:
            out["evidence"] = dict(self.evidence)
        return out


@dataclass(frozen=True)
class GateResult:
    verdict: Dict[str, Any]
    exit_code: int
    cells: Tuple[CellVerdict, ...] = ()

    @property
    def recommendation(self) -> str:
        return str(self.verdict.get("recommendation"))

    def verdict_for(self, channel: str, horizon: int) -> Optional[CellVerdict]:
        for cell in self.cells:
            if cell.key == (channel, horizon):
                return cell
        return None


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #


def _insufficient(
    channel: str, horizon: int, reason: str, detail: str, **evidence: Any
) -> CellVerdict:
    return CellVerdict(
        channel=channel, horizon=horizon, verdict=V_INSUFFICIENT, reason=reason,
        detail=detail, evidence=evidence,
    )


def _rejected(channel: str, horizon: int, reason: str, detail: str,
              **evidence: Any) -> CellVerdict:
    return CellVerdict(
        channel=channel, horizon=horizon, verdict=V_REJECT, reason=reason,
        detail=detail, evidence=evidence,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def _stderr_of(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    sd = _stdev(values)
    if sd == 0.0:
        return None
    return sd / math.sqrt(n)


def _derive_block(
    incumbent_errors: Sequence[float], candidate_errors: Sequence[float]
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Mean difference and its standard error, from paired per-row errors."""
    if len(incumbent_errors) != len(candidate_errors) or not incumbent_errors:
        return None, None, len(incumbent_errors)
    diffs = [float(i) - float(c) for i, c in zip(incumbent_errors, candidate_errors)]
    return _mean(diffs), _stderr_of(diffs), len(diffs)


def _half_reason(
    early: PairedBlock, late: PairedBlock
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Validate the two chronological halves, independently of their magnitude."""
    for key, block in (("early", early), ("late", late)):
        if not block.complete:
            return "%s_HALF_BLOCK_INCOMPLETE" % key.upper(), {}
        if block.n_pairs < MIN_HALF_PAIRED_ROWS:
            return "%s_HALF_TOO_FEW_ROWS" % key.upper(), {
                "%s_n_pairs" % key: block.n_pairs,
                "minimum": MIN_HALF_PAIRED_ROWS,
            }
        if block.stderr is not None and block.stderr <= 0.0:
            return "%s_HALF_STDERR_NOT_POSITIVE" % key.upper(), {}
        if block.mean_difference is not None and block.mean_difference <= 0.0:
            return "HALF_REGRESSION", {
                "%s_mean_difference" % key: block.mean_difference,
                "failing_half": key,
            }
    return None, {}


def _check_half_identities(
    full: PairedBlock, early: PairedBlock, late: PairedBlock
) -> Optional[Tuple[str, str]]:
    """The two halves must be disjoint, chronological, and exhaustive.

    Without this, "both halves" is a shape the caller can satisfy with one set
    of rows twice. That produces a confident, entirely vacuous pass.
    """
    # Row-count consistency is checkable from the counts alone, so it is checked
    # before and independently of any row identities. A half-split that does not
    # add up to the period it claims to split is not a split of that period.
    if (
        full.n_pairs is not None
        and early.n_pairs is not None
        and late.n_pairs is not None
        and early.n_pairs + late.n_pairs != full.n_pairs
    ):
        return "HALF_ROW_COUNTS_DO_NOT_SUM", (
            "early(%d) + late(%d) != full(%d); the two halves do not account for "
            "the validation period they claim to split"
            % (early.n_pairs, late.n_pairs, full.n_pairs)
        )

    ids_full, ids_early, ids_late = full.sample_ids, early.sample_ids, late.sample_ids
    if ids_full is None and ids_early is None and ids_late is None:
        return None
    if ids_early is None or ids_late is None:
        return "HALF_ROW_IDS_MISSING", (
            "sample ids are supplied for some spans but not all; the halves "
            "cannot be checked for overlap"
        )
    e, l = set(ids_early), set(ids_late)
    if e & l:
        return "HALVES_NOT_DISJOINT", (
            "%d row id(s) appear in both chronological halves, so the half-split "
            "check can be satisfied by one set of rows and proves nothing"
            % len(e & l)
        )
    if ids_full is not None:
        f = set(ids_full)
        if (e | l) != f:
            return "HALVES_DO_NOT_PARTITION_THE_PERIOD", (
                "the halves cover %d row(s) but the full period covers %d"
                % (len(e | l), len(f))
            )
    if list(ids_early) != sorted(ids_early) or list(ids_late) != sorted(ids_late):
        return "HALVES_NOT_CHRONOLOGICALLY_ORDERED", (
            "sample ids are not in chronological order within a half; the halves "
            "are not a time split"
        )
    if ids_early and ids_late and max(ids_early) > min(ids_late):
        return "HALVES_OVERLAP_IN_TIME", (
            "the 'early' half runs past the start of the 'late' half; this is not "
            "a chronological split of the validation period"
        )
    return None


def _check_supplied_against_rows(
    block: PairedBlock, computed_mean: Optional[float], computed_se: Optional[float]
) -> Optional[Tuple[str, str]]:
    if computed_mean is None:
        return None
    if block.mean_difference is None or abs(block.mean_difference - computed_mean) > 1e-9:
        return "PAIRED_STATISTIC_DISAGREEMENT", (
            "the supplied mean_difference (%r) does not match the mean of the "
            "supplied per-row errors (%r); rows are likely paired in the wrong "
            "order or the block describes a different span" % (block.mean_difference,
                                                              computed_mean)
        )
    if block.stderr is None or computed_se is None:
        return None
    if abs(block.stderr - computed_se) > 1e-9:
        return "PAIRED_STDERR_DISAGREEMENT", (
            "the supplied stderr (%r) does not match the standard error of the "
            "supplied per-row differences (%r)" % (block.stderr, computed_se)
        )
    return None


def decide_cell(
    channel: str,
    horizon: int,
    inputs: GateInputs,
    gate_blockers: Sequence[Tuple[str, str]],
) -> CellVerdict:
    """Apply the rule to one (channel, horizon) cell.

    Preconditions first, and they return rather than fall through: there is no
    ordering of this function in which missing evidence reaches a comparison.
    """
    spec = CHANNEL_SPECS.get(channel)
    if spec is None:
        return _insufficient(
            channel, horizon, "UNKNOWN_CHANNEL",
            "%r has no declared metric, direction, or row-error reduction; the "
            "gate will not guess which direction is better" % channel,
        )

    cand = dict(inputs.candidate).get((channel, horizon))
    inc = dict(inputs.incumbent).get((channel, horizon))
    if cand is None or inc is None:
        missing = "challenger" if cand is None else "incumbent"
        return _insufficient(
            channel, horizon, "CELL_ABSENT_FROM_EVIDENCE",
            "no %s evidence was supplied for this cell; an absent key is missing "
            "evidence, not a pass" % missing,
            metric=spec.metric,
        )

    for name, metric in (("candidate", cand.validation), ("incumbent", inc.validation)):
        if metric.value is None:
            return _insufficient(
                channel, horizon, "%s_METRIC_MISSING" % name.upper(),
                "the %s %s for this cell is %r; a missing metric can only ever "
                "be INSUFFICIENT_EVIDENCE"
                % (name, spec.metric, metric.value),
                metric=spec.metric, model=name,
            )

    paired = dict(cand.paired)
    full = paired.get("full")
    early = paired.get("early")
    late = paired.get("late")
    for key, block in (("full", full), ("early", early), ("late", late)):
        if block is None:
            return _insufficient(
                channel, horizon, "PAIRED_BLOCK_MISSING",
                "no %r paired block was supplied for this cell; the rule is a "
                "PAIRED test and cannot be evaluated from point estimates alone" % key,
                metric=spec.metric, missing_span=key,
            )
    if not full.complete:
        return _insufficient(
            channel, horizon, "PAIRED_BLOCK_INCOMPLETE",
            "the full-period paired block is missing n_pairs, mean_difference or "
            "stderr; all three are required",
            metric=spec.metric,
        )
    if full.n_pairs < MIN_PAIRED_ROWS:
        return _insufficient(
            channel, horizon, "PAIRED_ROWS_BELOW_MINIMUM",
            "%d paired rows, minimum %d" % (full.n_pairs, MIN_PAIRED_ROWS),
            metric=spec.metric, n_pairs=full.n_pairs,
            minimum=MIN_PAIRED_ROWS,
        )
    if full.stderr <= 0.0:
        return _insufficient(
            channel, horizon, "PAIRED_STDERR_NOT_POSITIVE",
            "the paired standard error is %r; with no spread there is no "
            "threshold to clear and the result is not measurable, not equal"
            % full.stderr,
            metric=spec.metric, stderr=full.stderr,
        )

    problem = _check_half_identities(full, early, late)
    if problem is not None:
        return _insufficient(
            channel, horizon, problem[0], problem[1],
            metric=spec.metric,
        )
    # Cross-check a supplied paired block against its own per-row errors. This
    # is the only place the gate can notice that rows were paired in the wrong
    # order -- a mistake that produces a confident, wrong, well-formatted
    # verdict rather than an absence. It runs before the verdict, because a
    # statistic that disagrees with its own rows cannot be trusted for anything,
    # including being called worse.
    if full.incumbent_errors is not None and full.candidate_errors is not None:
        cmean, cse, cn = _derive_block(full.incumbent_errors, full.candidate_errors)
        if cmean is None or cn != full.n_pairs:
            return _insufficient(
                channel, horizon, "PAIRED_ROW_COUNT_MISMATCH",
                "the per-row error vectors hold %d pairs but the block declares "
                "%s" % (cn, full.n_pairs),
                metric=spec.metric, derived_pairs=cn,
                declared_pairs=full.n_pairs,
            )
        problem = _check_supplied_against_rows(full, cmean, cse)
        if problem is not None:
            return _insufficient(
                channel, horizon, problem[0], problem[1],
                metric=spec.metric,
            )

    # Direction first. A challenger that is behind over the whole validation
    # period is reported as WORSE, which is the actionable description; calling
    # it a "half regression" would point the reader at one half of a loss they
    # can see everywhere.
    if full.mean_difference < 0.0:
        return _rejected(
            channel, horizon, REASON_WORSE,
            "the challenger is worse over the whole validation period: mean "
            "paired difference %r (challenger better is positive)"
            % full.mean_difference,
            metric=spec.metric, n_pairs=full.n_pairs,
            mean_difference=full.mean_difference, stderr=full.stderr,
        )

    reason, extra = _half_reason(early, late)
    if reason is not None:
        if reason == "HALF_REGRESSION":
            return _rejected(
                channel, horizon, REASON_HALF_REGRESSION,
                "the challenger is not ahead in the %s half of the validation "
                "period; a win concentrated in one half is not a win"
                % extra.get("failing_half", "?"),
                metric=spec.metric, **extra
            )
        return _insufficient(
            channel, horizon, reason,
            "the chronological half-split could not be evaluated",
            metric=spec.metric, **extra
        )

    if gate_blockers:
        name, detail = gate_blockers[0]
        return _insufficient(
            channel, horizon, "GATE_INTEGRITY_BLOCKED",
            "%s: %s" % (name, detail),
            metric=spec.metric, blocker=name,
        )

    served = None if inputs.policy is None else inputs.policy.resolve(channel, horizon)
    if served is None:
        why = "the policy view is unavailable" if inputs.policy is None \
            else (inputs.policy.read_error or "the cell is absent from the policy")
        return _insufficient(
            channel, horizon, "POLICY_SOURCE_UNKNOWN",
            "inference_policy.json does not say what is served for %s at +%dh (%s); "
            "promoting into an unknown source is not a decision" % (channel, horizon, why),
            metric=spec.metric,
        )

    required = SIGMA_MULTIPLIER * full.stderr
    cand_v = float(cand.validation.value)
    inc_v = float(inc.validation.value)
    denom = abs(inc_v) if abs(inc_v) > 1e-12 else None
    evidence: Dict[str, Any] = {
        "metric": spec.metric,
        "unit": spec.unit,
        "better": spec.better,
        "row_error": spec.row_error,
        "decision_split": SPLIT_DECISION,
        "incumbent": inc_v,
        "challenger": cand_v,
        "absolute_improvement": inc_v - cand_v,
        "relative_improvement": (
            None if denom is None else (inc_v - cand_v) / denom
        ),
        "n_pairs": full.n_pairs,
        "mean_difference": full.mean_difference,
        "stderr": full.stderr,
        "required_difference": required,
        "sigma": (None if full.stderr == 0.0 else full.mean_difference / full.stderr),
        "halves": {
            "early": {
                "n_pairs": early.n_pairs,
                "mean_difference": early.mean_difference,
                "stderr": early.stderr,
                "ahead": bool(early.mean_difference > 0.0),
            },
            "late": {
                "n_pairs": late.n_pairs,
                "mean_difference": late.mean_difference,
                "stderr": late.stderr,
                "ahead": bool(late.mean_difference > 0.0),
            },
        },
        "stderr_source": (
            "COMPUTED_FROM_ROWS" if full.incumbent_errors is not None else "SUPPLIED"
        ),
        "evidence_sources": {
            span: block.source for span, block in sorted(paired.items()) if block.source
        },
        "incumbent_source": served.source,
        "incumbent_source_detail": served.detail,
        "promotion_class": (
            BEATS_PERSISTENCE if served.source in PERSISTENCE_SOURCES
            else BEATS_SERVED_MODEL
        ),
    }

    if full.mean_difference <= required:
        # Every negative mean difference has already returned REASON_WORSE
        # above, so reaching here means "not worse, but not convincingly
        # better". A second WORSE branch here would be unreachable code, and
        # unreachable branches in a gate are how the two drift apart.
        return _rejected(
            channel, horizon, REASON_NOT_SIGNIFICANT,
            "mean paired difference %r does not exceed %g sigma (%r required)"
            % (full.mean_difference, SIGMA_MULTIPLIER, required),
            **evidence
        )

    return CellVerdict(
        channel=channel, horizon=horizon, verdict=V_PROMOTE, reason=REASON_PROMOTED,
        detail="mean paired difference %r exceeds %g sigma (%r) and the "
               "challenger is ahead in both chronological halves"
               % (full.mean_difference, SIGMA_MULTIPLIER, required),
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Gate-level integrity
# --------------------------------------------------------------------------- #


def _integrity(inputs: GateInputs) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """Checks that gate the WHOLE run, evaluated before any cell.

    Each entry that is not PASS appends a blocker. A blocker propagates into
    every cell as INSUFFICIENT_EVIDENCE, so a run on the wrong corpus cannot
    produce a promotion for any channel.
    """
    blockers: List[Tuple[str, str]] = []
    report: Dict[str, Any] = {}

    candidate_sha = (inputs.corpus_sha256 or "").strip()
    baseline_sha = None if inputs.baseline is None else inputs.baseline.corpus_sha256
    policy_sha = None if inputs.policy is None else inputs.policy.weather_telemetry_sha256

    corpus: Dict[str, Any] = {
        "candidate": candidate_sha or None,
        "baseline": baseline_sha,
        "policy": policy_sha,
    }
    if not candidate_sha:
        corpus["status"] = "FAIL"
        corpus["reason"] = "CORPUS_SHA_MISSING"
        blockers.append((
            "CORPUS_SHA_MISSING",
            "no corpus SHA-256 was supplied for the challenger, so the candidate "
            "and the incumbent cannot be shown to have been scored on the same "
            "data",
        ))
    elif baseline_sha is None:
        corpus["status"] = "FAIL"
        corpus["reason"] = "BASELINE_CORPUS_UNKNOWN"
        blockers.append((
            "BASELINE_CORPUS_UNKNOWN",
            "the incumbent record carries no corpus SHA-256, so there is no way "
            "to show the comparison is on identical data",
        ))
    elif candidate_sha != baseline_sha:
        corpus["status"] = "FAIL"
        corpus["reason"] = "CORPUS_MISMATCH"
        blockers.append((
            "CORPUS_MISMATCH",
            "the challenger was scored on corpus %s and the incumbent record was "
            "captured on %s; a difference between two models scored on different "
            "data is not a difference between the models"
            % (candidate_sha[:16], baseline_sha[:16]),
        ))
    else:
        corpus["status"] = "PASS"
    report["corpus"] = corpus

    if policy_sha and baseline_sha and policy_sha != baseline_sha:
        report["policy_corpus"] = {
            "status": "FAIL",
            "reason": "POLICY_CORPUS_DIVERGED",
            "policy": policy_sha,
            "baseline": baseline_sha,
        }
        blockers.append((
            "POLICY_CORPUS_DIVERGED",
            "inference_policy.json was refitted against corpus %s but the "
            "baseline was captured on %s; the served-source selection and the "
            "recorded scores describe different data"
            % (policy_sha[:16], baseline_sha[:16]),
        ))
    else:
        report["policy_corpus"] = {"status": "PASS"}

    if inputs.policy is None:
        report["policy"] = {"status": "FAIL", "reason": "POLICY_UNAVAILABLE"}
        blockers.append((
            "POLICY_UNAVAILABLE",
            "no inference_policy source selection was supplied; the gate cannot "
            "state what is currently being served for any cell",
        ))
    else:
        pdict = inputs.policy.to_dict()
        if inputs.policy.read_error:
            pdict["status"] = "FAIL"
            pdict["reason"] = "POLICY_UNREADABLE"
            blockers.append((
                "POLICY_UNREADABLE",
                "inference_policy.json could not be read (%s)"
                % inputs.policy.read_error,
            ))
        elif not pdict.get("cells_resolved"):
            pdict["status"] = "FAIL"
            pdict["reason"] = "POLICY_EMPTY"
            blockers.append((
                "POLICY_EMPTY",
                "inference_policy.json resolved no served sources at all",
            ))
        else:
            pdict["status"] = "PASS"
        report["policy"] = pdict

    if inputs.baseline is not None and inputs.baseline.parse_errors:
        report["baseline_document"] = {
            "status": "FAIL",
            "reason": "BASELINE_PARSE_ERRORS",
            "parse_errors": list(inputs.baseline.parse_errors),
        }
        blockers.append((
            "BASELINE_PARSE_ERRORS",
            "the incumbent record did not parse cleanly: %s"
            % "; ".join(inputs.baseline.parse_errors),
        ))
    else:
        report["baseline_document"] = {"status": "PASS"}

    drift = inputs.drift
    if drift is None:
        report["baseline_drift"] = {
            "status": "NOT_CHECKED",
            "reason": "DRIFT_CHECK_NOT_SUPPLIED",
        }
        blockers.append((
            "DRIFT_CHECK_NOT_SUPPLIED",
            "no baseline re-scoring was supplied; the incumbent record has not "
            "been checked for drift, so every comparison is against a fixed "
            "point nobody has verified",
        ))
    else:
        dd = drift.to_dict()
        if drift.status == DRIFT_MATCH:
            dd["status"] = "PASS"
        elif drift.status == DRIFT_DRIFT:
            dd["reason"] = "BASELINE_DRIFT"
            blockers.append((
                "BASELINE_DRIFT",
                "a fresh scoring of the incumbent disagrees with the recorded "
                "record for %d cell(s) (%s); the baseline is stale"
                % (len(drift.drifted), ", ".join("%s|h%d" % k for k in drift.drifted)),
            ))
        elif drift.status == DRIFT_NOT_CHECKED:
            dd["reason"] = "BASELINE_NOT_RECHECKED"
            blockers.append((
                "BASELINE_NOT_RECHECKED",
                "the incumbent re-scoring produced no values; the record is "
                "unverified for every cell it covers",
            ))
        else:
            dd["reason"] = "BASELINE_RECHECK_SKIPPED"
            blockers.append((
                "BASELINE_RECHECK_SKIPPED",
                "the incumbent re-scoring was skipped (%s); a guard that can be "
                "skipped cannot catch a corpus or scoring change that moved "
                "underneath the incumbent"
                % (drift.justification or "no justification recorded"),
            ))
        report["baseline_drift"] = dd

    return blockers, report


def _recommendation(cells: Sequence[CellVerdict]) -> Tuple[str, int]:
    counts = {V_PROMOTE: 0, V_REJECT: 0, V_INSUFFICIENT: 0}
    for cell in cells:
        counts[cell.verdict] = counts.get(cell.verdict, 0) + 1
    if counts[V_INSUFFICIENT]:
        return R_BLOCKED, EXIT_BLOCKED
    if counts[V_PROMOTE] and counts[V_REJECT]:
        return R_SPLIT, EXIT_OK
    if counts[V_PROMOTE]:
        return R_PROMOTE_ALL, EXIT_OK
    return R_RETAIN_ALL, EXIT_RETAIN


def evaluate_promotion(
    inputs: GateInputs,
) -> Tuple[GateResult, int]:
    """Run the gate.

    Returns ``(result, exit_code)``. The exit code is returned rather than
    buried in the result so a caller cannot read a verdict and lose the fact
    that the run was blocked.
    """
    try:
        blockers, integrity = _integrity(inputs)
        cells = [decide_cell(ch, h, inputs, blockers) for ch, h in inputs.cells()]
    except GateInputError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise GateInputError("gate could not evaluate: %s: %s"
                             % (type(exc).__name__, exc))

    recommendation, exit_code = _recommendation(cells)
    counts: Dict[str, int] = {V_PROMOTE: 0, V_REJECT: 0, V_INSUFFICIENT: 0}
    for cell in cells:
        counts[cell.verdict] += 1
    for cell in cells:
        assert cell.verdict in counts, "verdict %r is outside the vocabulary" % cell.verdict
        if cell.verdict != V_PROMOTE:
            assert cell.reason, (
                "a non-promotion for %s|h%d carries no reason; an unexplained "
                "verdict is the silent-failure shape this gate exists to prevent"
                % (cell.channel, cell.horizon)
            )

    promoted = ["%s|h%d" % (c.channel, c.horizon)
                for c in cells if c.verdict == V_PROMOTE]
    rejected = ["%s|h%d" % (c.channel, c.horizon)
                for c in cells if c.verdict == V_REJECT]
    insufficient = ["%s|h%d" % (c.channel, c.horizon)
                    for c in cells if c.verdict == V_INSUFFICIENT]
    beats_persistence = [
        "%s|h%d" % (c.channel, c.horizon) for c in cells
        if c.verdict == V_PROMOTE
        and c.evidence.get("promotion_class") == BEATS_PERSISTENCE
    ]
    beats_model = [k for k in promoted if k not in set(beats_persistence)]

    verdict = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": "prediction-model/src/release_gate.py",
        "recommendation": recommendation,
        "promotion_decision_computed": recommendation != R_BLOCKED,
        "exit_code": exit_code,
        "exit_code_meaning": EXIT_MEANING[exit_code],
        "decision_inputs_used": {
            "corpus_sha256": inputs.corpus_sha256,
            "channels": list(inputs.channels),
            "horizons": [int(h) for h in inputs.horizons],
            "split_used_for_decision": SPLIT_DECISION,
            "split_used_for_reporting": SPLIT_REPORTING,
            "incumbent": inputs.incumbent_name,
            "challenger": inputs.candidate_name,
        },
        "rule": RULE,
        "integrity": integrity,
        "counts": counts,
        "summary": {
            "promoted": promoted,
            "rejected": rejected,
            "insufficient_evidence": insufficient,
            "promotions_beating_persistence": beats_persistence,
            "promotions_beating_served_model": beats_model,
        },
        "cells": {"%s|h%d" % (c.channel, c.horizon): c.to_dict() for c in cells},
    }
    return GateResult(verdict=verdict, exit_code=exit_code, cells=tuple(cells)), exit_code


# --------------------------------------------------------------------------- #
# JSON input / output
# --------------------------------------------------------------------------- #


def _metric_from_json(node: Any, label: str) -> CellMetric:
    if node is None:
        return CellMetric(value=None, split=SPLIT_DECISION, source=label)
    if not isinstance(node, dict):
        raise GateInputError("%s is not an object" % label)
    return CellMetric(
        value=_optional_number(node.get("value"), "%s.value" % label),
        n_rows=node.get("n_rows"),
        split=node.get("split", SPLIT_DECISION),
        source=node.get("source", label),
    )


def _block_from_json(node: Any, label: str) -> Optional[PairedBlock]:
    if node is None:
        return None
    if not isinstance(node, dict):
        raise GateInputError("%s is not an object" % label)
    ids = node.get("sample_ids")
    inc = node.get("incumbent_errors")
    cand = node.get("candidate_errors")
    return PairedBlock(
        n_pairs=node.get("n_pairs"),
        mean_difference=node.get("mean_difference"),
        stderr=node.get("stderr"),
        split=node.get("split", SPLIT_DECISION),
        source=node.get("source", label),
        sample_ids=None if ids is None else tuple(ids),
        incumbent_errors=None if inc is None else tuple(inc),
        candidate_errors=None if cand is None else tuple(cand),
    )


def _evidence_from_json(node: Any, label: str) -> CellEvidence:
    if node is None:
        return CellEvidence()
    if not isinstance(node, dict):
        raise GateInputError("%s is not an object" % label)
    paired: Dict[str, PairedBlock] = {}
    for span, block in (node.get("paired") or {}).items():
        parsed = _block_from_json(block, "%s.paired.%s" % (label, span))
        if parsed is not None:
            paired[str(span)] = parsed
    test_node = node.get("test")
    if test_node is not None and not isinstance(test_node, dict):
        raise GateInputError("%s.test is not an object" % label)
    return CellEvidence(
        validation=_metric_from_json(node.get("validation"), "%s.validation" % label),
        test=(
            None
            if test_node is None
            else CellMetric(
                value=_optional_number(test_node.get("value"), "%s.test.value" % label),
                n_rows=test_node.get("n_rows"),
                split=test_node.get("split", SPLIT_REPORTING),
                source=test_node.get("source", "%s.test" % label),
            )
        ),
        paired=paired,
    )


def load_gate_inputs(path: str) -> GateInputs:
    """Read a JSON decision-input bundle.

    A missing key inside a cell means the metric is missing. It is NOT filled
    with a default, and it is NOT dropped from the grid.
    """
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise GateInputError("%s is not a JSON object" % path)

    def _cells(section: Any, label: str) -> Dict[Tuple[str, int], CellEvidence]:
        out: Dict[Tuple[str, int], CellEvidence] = {}
        if not isinstance(section, dict):
            return out
        for key, node in section.items():
            channel, _, horizon = str(key).partition("|")
            h = horizon.lstrip("hH")
            try:
                out[(channel, int(h))] = _evidence_from_json(node, "%s.%s" % (label, key))
            except ValueError:
                raise GateInputError(
                    "%s.%s: horizon key must look like '<channel>|<h><hours>'" % (label, key)
                )
        return out

    baseline_path = doc.get("baseline_path")
    policy_path = doc.get("policy_path")
    baseline = None
    if isinstance(baseline_path, str):
        baseline = load_release_baseline(baseline_path)
    elif isinstance(doc.get("baseline"), dict):
        baseline = load_release_baseline(
            doc["baseline"].get("path", DEFAULT_BASELINE_PATH)
        )

    policy = None
    if isinstance(doc.get("policy"), dict):
        policy = load_policy(doc["policy"].get("path", DEFAULT_POLICY_PATH))
    elif isinstance(policy_path, str):
        policy = load_policy(policy_path)

    drift_node = doc.get("baseline_drift")
    drift: Optional[DriftReport] = None
    if isinstance(drift_node, dict) and drift_node.get("status") == DRIFT_SKIPPED:
        drift = DriftReport(
            status=DRIFT_SKIPPED,
            tolerance_rel=drift_node.get("tolerance_rel", DRIFT_REL_TOLERANCE),
            justification=drift_node.get("justification"),
        )

    fresh = doc.get("fresh_incumbent_metrics")
    if isinstance(fresh, dict):
        fresh_map: Optional[Dict[Tuple[str, int], Optional[float]]] = {}
        for key, value in fresh.items():
            channel, _, horizon = str(key).partition("|")
            fresh_map[(channel, int(horizon.lstrip("hH")))] = (
                None if value is None else _optional_number(value, "fresh[%s]" % key)
            )
        target = baseline if baseline is not None else None
        if target is not None:
            drift = compare_baseline_to_fresh(
                target,
                fresh_map,
                tolerance_rel=float(doc.get("drift_tolerance_rel", DRIFT_REL_TOLERANCE)),
                justification=drift_node.get("justification")
                if isinstance(drift_node, dict) else None,
            )
    if drift is None and isinstance(drift_node, dict):
        raise GateInputError(
            "baseline_drift was supplied as a report but without a fresh scoring "
            "to compute it from; supply 'fresh_incumbent_metrics' or an explicit "
            "skipped report carrying a justification"
        )

    return GateInputs(
        corpus_sha256=doc.get("corpus_sha256", ""),
        candidate=_cells(doc.get("challenger"), "challenger"),
        incumbent=_cells(doc.get("incumbent"), "incumbent"),
        channels=tuple(doc.get("channels", DEFAULT_CHANNELS)),
        horizons=tuple(doc.get("horizons", HORIZONS)),
        baseline=baseline,
        policy=policy,
        drift=drift,
        candidate_name=str(doc.get("challenger_name", "challenger")),
        incumbent_name=str(doc.get("incumbent_name", "incumbent")),
    )


def write_verdict(result: GateResult, path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "release_gate_verdict.json")
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result.verdict, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def summarize(result: GateResult) -> str:
    v = result.verdict
    s = v["summary"]
    lines = [
        "release gate: %s (exit %d)" % (v["recommendation"], result.exit_code),
        "  rule: %.1f-sigma paired, both validation halves, per channel; "
        "min pairs=%d" % (v["rule"]["sigma_multiplier"],
                          v["rule"]["min_paired_rows"]),
        "  cells: %d promote, %d reject, %d insufficient evidence"
        % (v["counts"][V_PROMOTE], v["counts"][V_REJECT], v["counts"][V_INSUFFICIENT]),
        "  promoted: %s" % (", ".join(s["promoted"]) or "none"),
        "  rejected: %s" % (", ".join(s["rejected"]) or "none"),
    ]
    if s["insufficient_evidence"]:
        lines.append("  no decision (INSUFFICIENT_EVIDENCE): %s"
                     % ", ".join(s["insufficient_evidence"]))
    integrity = v["integrity"]
    bad = [k for k, val in integrity.items() if isinstance(val, dict)
           and val.get("status") not in ("PASS", None)]
    if bad:
        lines.append("  INTEGRITY BLOCKERS: %s" % ", ".join(
            "%s (%s)" % (k, integrity[k].get("reason", "fail")) for k in bad))
    if v["recommendation"] == R_BLOCKED:
        lines.append("  promotion was NOT evaluated. This is not a pass.")
    if s["promotions_beating_persistence"]:
        lines.append(
            "  NOTE: %d promotion(s) beat persistence, not a served model: %s"
            % (len(s["promotions_beating_persistence"]),
               ", ".join(s["promotions_beating_persistence"]))
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="release_gate.py",
        description=(
            "Decide whether a challenger may replace the incumbent, per channel "
            "and per horizon, on validation data only. Exits non-zero whenever "
            "it cannot make a decision."
        ),
    )
    ap.add_argument("--inputs", required=True,
                    help="JSON bundle of decision inputs (see load_gate_inputs)")
    ap.add_argument("--out", required=True,
                    help="path for the JSON verdict document")
    ap.add_argument("--quiet", action="store_true", help="suppress the summary")
    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        inputs = load_gate_inputs(args.inputs)
        result, code = evaluate_promotion(inputs)
    except GateInputError as exc:
        sys.stderr.write("UNUSABLE INPUT: %s\n" % exc)
        sys.stderr.write(
            "the gate did not run; no promotion decision exists\n"
        )
        return EXIT_UNUSABLE
    except (OSError, ValueError) as exc:
        sys.stderr.write("UNUSABLE INPUT: %s: %s\n" % (type(exc).__name__, exc))
        sys.stderr.write("the gate did not run; no promotion decision exists\n")
        return EXIT_UNUSABLE

    path = write_verdict(result, args.out)
    if not args.quiet:
        sys.stdout.write(summarize(result) + "\n")
        sys.stdout.write("  verdict written: %s\n" % path)
    return code


if __name__ == "__main__":
    sys.exit(main())
