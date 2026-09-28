# Source License Decisions

## Purpose

This document records the license and data-rights review decisions for all
candidate external sources considered for the prediction model.

A source may enter production only when its registry decision is one of:

- `APPROVED_FREE_COMMERCIAL`
- `APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION`

All other states are blocked from production use.

---

## Decision Summary

| Source ID | Provider | Decision | Reason |
|---|---|---|---|
| `project_nearby_stations_v1` | Project-owned/partner | `UNKNOWN_BLOCKED` | No data-sharing agreement reviewed |
| `radar_qpe_v1` | Unknown radar provider | `UNKNOWN_BLOCKED` | No specific product identified or reviewed |
| `rainviewer_v1` | RainViewer | `UNKNOWN_BLOCKED` | Public API ≠ commercial license |
| `himawari9_derived_v1` | JMA/BoM/NCI | `UNKNOWN_BLOCKED` | Product distribution path unconfirmed |
| `pagasa_nwp_v1` | PAGASA | `UNKNOWN_BLOCKED` | Product-specific usage rights unconfirmed |

---

## Detailed Reviews

### Candidate A — Project-Owned Nearby Stations

- **Source ID**: `project_nearby_stations_v1`
- **Decision**: `UNKNOWN_BLOCKED`
- **Value**: High — provides regional gradients for temperature, pressure, humidity,
  wind without image processing or complex product interpretation.
- **Blocker**: No station partnership or data-sharing agreement has been reviewed.
  Station ownership, telemetry format, availability guarantees, and commercial
  use rights must be confirmed.
- **Action Required**: Identify specific stations, confirm ownership or partnership,
  document data-sharing terms, and update the registry.

### Candidate B — Radar/QPE

- **Source ID**: `radar_qpe_v1`
- **Decision**: `UNKNOWN_BLOCKED`
- **Value**: High for rain onset, cessation, and heavy-rain anomalies.
- **Blocker**: No specific radar product has been identified. The exact provider,
  product name, data access method, and commercial rights must be documented.
- **Action Required**: Identify the specific radar QPE product, obtain license
  terms, confirm commercial training and inference rights, and update the registry.

### RainViewer

- **Source ID**: `rainviewer_v1`
- **Decision**: `UNKNOWN_BLOCKED`
- **Value**: Convenient radar coverage, but not automatically acceptable for
  commercial production.
- **Blocker**: Public API access does not constitute a commercial license. The
  Terms of Use have not been reviewed for:
  - Commercial use in model training
  - Commercial inference
  - Derived feature creation
  - Caching and redistribution
- **Important**: "A public endpoint is not a commercial license."
  (Section 16 of the implementation plan.)
- **Action Required**: Review RainViewer Terms of Use in detail. Contact RainViewer
  if commercial use requires a paid plan or explicit permission.

### Candidate C — Himawari-9 Derived Features

- **Source ID**: `himawari9_derived_v1`
- **Decision**: `UNKNOWN_BLOCKED`
- **Value**: Potential for cloud growth, convection, and radiation context.
- **Blocker**: The exact product distribution path (JMA direct, BoM mirror, NCI
  archive, or other) and its commercial training rights have not been confirmed.
  Different distribution paths may have different license terms.
- **Action Required**: Identify the exact data access path, review the specific
  license terms for that path, confirm commercial training and inference rights,
  and update the registry.

### Candidate D — PAGASA NWP

- **Source ID**: `pagasa_nwp_v1`
- **Decision**: `UNKNOWN_BLOCKED`
- **Value**: Potentially valuable as a regional prior at 6h–24h horizons.
- **Blocker**: Product-specific usage rights and issue-time replay capability
  have not been confirmed. Government data may have specific terms for commercial
  use.
- **Action Required**: Contact PAGASA or review published data policies, confirm
  commercial use terms, verify historical availability for replay, and update
  the registry.

---

## Current System Impact

Since no external source has been approved:

1. **The prediction model operates in local-only mode.**
2. All external source adapter stubs are created but cannot be used in production.
3. The license gate in `external_source_registry.py` blocks all sources.
4. This is a valid scientific outcome, not a failure of the release process.

## Next Steps

To unblock external source integration:

1. Select the highest-value, lowest-risk candidate (Candidate A preferred).
2. Complete the rights review documented in the action items above.
3. Update `external_source_registry.json` with the review results.
4. If approved, change the decision to `APPROVED_FREE_COMMERCIAL` or
   `APPROVED_WITH_EXPLICIT_NO_COST_PERMISSION` and fill in all required fields.
5. Run `assert_production_eligible()` to verify the gate passes.
6. Proceed with adapter integration for the approved source only.
