"use client";

/**
 * prediction-provenance-panel.tsx
 *
 * An operator's answer to "where did this number come from, and should I trust
 * it?" for any single displayed forecast.
 *
 * ── WHAT THIS PANEL IS FOR ──────────────────────────────────────────────────
 * The Python side has a strong audit trail: `prediction_audit.py` records which
 * model, checkpoint, policy and producer produced every published value, and
 * `verify_predictions.py` scores matured forecasts against observations that
 * have since arrived. None of that reached an operator. This panel is the read
 * path for it.
 *
 * ── FIVE ANSWERS, IN ORDER OF IMPORTANCE ────────────────────────────────────
 *  1. Which model and checkpoint, at what time, from which policy commit.
 *  2. Was this cell actually predicted, or was the origin observation carried
 *     forward? At a glance, before anything else on the screen is read.
 *  3. Which data it used, and whether the router found anything usable.
 *  4. The station and timestamp of the observation behind any persistence value.
 *  5. How accurate this model has actually been for THIS variable, where
 *     verification exists -- and an explicit statement that it does not, where
 *     it does not.
 *
 * ── HONESTY CONSTRAINTS BAKED INTO THE RENDER ───────────────────────────────
 *  - No pooled MAE anywhere. Errors are grouped by variable, each with its own
 *    unit, matching the deliberate absence of a cross-unit aggregate in
 *    `verify_predictions.py`.
 *  - Verified (scored against real observations) and expected (a static offline
 *    benchmark) live in separate sections with separate headings. They are never
 *    combined into one figure.
 *  - A blank accuracy cell is rendered with words ("no skill over persistence",
 *    "not measured"), never a 0, and never an em dash that could read as zero.
 *  - When the recorded producer label disagrees with the inference policy, both
 *    are shown and the disagreement is counted. The panel does not pick a
 *    winner.
 *
 * ── STYLE ───────────────────────────────────────────────────────────────────
 * Dense and operational, like the rest of the internal tooling: monospace,
 * `tabular-nums`, 10-11px, slate palette, no hero type, no governance banner.
 * `ModelGovernanceBanner` already owns the research-prototype disclaimer on the
 * prediction surface; repeating it here would be noise.
 *
 * It is not wired into any page. See ./README-endpoint.md for the endpoint shape
 * it expects and the one-line change needed to mount it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Database, Loader2, RefreshCw } from "lucide-react";

import {
  AUDITED_VARIABLES,
  VARIABLE_UNITS,
  assessCarryForward,
  buildExpectedAccuracyTable,
  buildSourceMatrix,
  buildVerifiedAccuracyTable,
  conversionNote,
  describeCellGap,
  describeCoverage,
  formatMetric,
  formatMinutes,
  formatNumber,
  formatUtc,
  minutesBetween,
  selectForecastRecord,
  shortSha,
  skillOverPersistence,
  type AuditRecord,
  type CarryForwardVerdict,
  type CellClassification,
  type ExpectedCellStatus,
  type ProvenancePayload,
  type VariableName,
} from "./prediction-provenance-core";
import { buildExpectedAccuracySection } from "./expected-accuracy.reference";
import {
  DEFAULT_PROVENANCE_ENDPOINT,
  ProvenanceFetchError,
  fetchProvenancePayload,
} from "./provenance-api";

/* ────────────────────────────────────────────────────────────────────────────
 * Props
 * ──────────────────────────────────────────────────────────────────────────── */

export interface PredictionProvenancePanelProps {
  stationId: string;
  /** Shown next to the station id; purely an operator label. */
  stationName?: string;
  /** Horizon currently displayed on the calling surface. */
  horizonHours?: number;
  onSelectHorizon?: (horizonHours: number) => void;
  /**
   * Pre-loaded payload. Supplied by a server component that already has the
   * data; when present the panel does not fetch.
   */
  initialPayload?: ProvenancePayload | null;
  endpoint?: string;
  refreshIntervalMs?: number;
  className?: string;
}

/* ────────────────────────────────────────────────────────────────────────────
 * Compact presentational atoms
 * ──────────────────────────────────────────────────────────────────────────── */

const CELL_BASE =
  "border border-slate-800/70 bg-slate-950/40 px-2 py-1 font-mono text-[11px] leading-tight";

function Cell({ children, className = "", title }: { children?: React.ReactNode; className?: string; title?: string }) {
  return (
    <div className={`${CELL_BASE} ${className}`} title={title}>
      {children}
    </div>
  );
}

function SectionLabel({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-slate-800 pb-1">
      <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">
        {children}
      </h3>
      {right ? <span className="font-mono text-[10px] text-slate-500">{right}</span> : null}
    </div>
  );
}

/** The one visual that has to read at a glance across a room. */
function ServedByTag({ servedBy }: { servedBy: CellClassification["servedBy"] }) {
  if (servedBy === "MODEL") {
    return (
      <span className="inline-block border border-emerald-700/70 bg-emerald-950/60 px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase text-emerald-300">
        model
      </span>
    );
  }
  if (servedBy === "PERSISTENCE") {
    return (
      <span
        className="inline-block border border-amber-600/80 bg-amber-950/60 px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase text-amber-300"
        title="Carried forward from the origin observation. Not a prediction."
      >
        persistence
      </span>
    );
  }
  if (servedBy === "DERIVED") {
    return (
      <span className="inline-block border border-sky-800/80 bg-sky-950/60 px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase text-sky-300">
        derived
      </span>
    );
  }
  return (
    <span className="inline-block border border-slate-700 bg-slate-900 px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase text-slate-400">
      unknown
    </span>
  );
}

function ConflictMark({ reason }: { reason: string | null }) {
  if (!reason) return <span className="font-mono text-[10px] text-slate-600">—</span>;
  return (
    <span className="inline-flex items-center gap-1 font-mono text-[10px] text-rose-300" title={reason}>
      <AlertTriangle className="h-3 w-3 shrink-0" aria-hidden />
      conflict
    </span>
  );
}

function Gap({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <span className="font-mono text-[10px] italic text-slate-500" title={title}>
      {children}
    </span>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Main component
 * ──────────────────────────────────────────────────────────────────────────── */

export default function PredictionProvenancePanel({
  stationId,
  stationName,
  horizonHours,
  onSelectHorizon,
  initialPayload = null,
  endpoint = DEFAULT_PROVENANCE_ENDPOINT,
  refreshIntervalMs = 0,
  className = "",
}: PredictionProvenancePanelProps) {
  const [payload, setPayload] = useState<ProvenancePayload | null>(initialPayload);
  const [selectedHorizon, setSelectedHorizon] = useState<number | null>(horizonHours ?? null);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    initialPayload ? "ready" : "idle"
  );
  // Kept separate from `status` because a refresh that lands while an earlier
  // payload is still on screen must show as in-flight, not as an error.
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<ProvenanceFetchError | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (typeof horizonHours === "number") setSelectedHorizon(horizonHours);
  }, [horizonHours]);

  const load = useCallback(
    (signal: AbortSignal) => {
      setStatus((current) => (current === "ready" ? "ready" : "loading"));
      setRefreshing(true);
      setError(null);
      return fetchProvenancePayload(
        {
          stationId,
          horizonHours: selectedHorizon,
          recordId: selectedRecordId,
        },
        { endpoint, signal }
      )
        .then((next) => {
          setPayload(next);
          setStatus("ready");
        })
        .catch((err: unknown) => {
          if (err instanceof ProvenanceFetchError && err.kind === "aborted") return;
          setError(
            err instanceof ProvenanceFetchError
              ? err
              : new ProvenanceFetchError("network", String((err as Error)?.message ?? err))
          );
          setStatus("error");
        })
        .finally(() => setRefreshing(false));
    },
    [endpoint, selectedHorizon, selectedRecordId, stationId]
  );

  useEffect(() => {
    if (initialPayload) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    void load(controller.signal);
    return () => controller.abort();
  }, [initialPayload, load, reloadToken]);

  useEffect(() => {
    if (!refreshIntervalMs || refreshIntervalMs <= 0 || initialPayload) return;
    const timer = setInterval(() => setReloadToken((n) => n + 1), refreshIntervalMs);
    return () => clearInterval(timer);
  }, [initialPayload, refreshIntervalMs]);

  const handleHorizon = useCallback(
    (hours: number) => {
      setSelectedHorizon(hours);
      setSelectedRecordId(null);
      onSelectHorizon?.(hours);
    },
    [onSelectHorizon]
  );

  /* ── derivations ─────────────────────────────────────────────────────── */

  /**
   * The record under inspection.
   *
   * An explicit `selectedRecordId` always wins, because "show me this exact
   * published number" is the question the panel exists to answer. Otherwise the
   * server's chosen `payload.forecast` is used, and only if the endpoint did
   * not pick one does the client select from history itself.
   */
  const forecast = useMemo(() => {
    if (!payload) return null;
    if (selectedRecordId) {
      const pinned = payload.forecastHistory?.find(
        (record) => record.record_id === selectedRecordId
      );
      if (pinned) return pinned;
      if (payload.forecast?.record_id === selectedRecordId) return payload.forecast;
    }
    const pool = payload.forecastHistory?.length
      ? payload.forecastHistory
      : payload.forecast
        ? [payload.forecast]
        : [];
    return (
      payload.forecast ??
      selectForecastRecord(pool, { stationId, horizonHours: selectedHorizon })
    );
  }, [payload, selectedHorizon, selectedRecordId, stationId]);

  const observedProducers = useMemo(() => {
    const out: Record<string, Record<string, string | null>> = {};
    const pool = payload?.forecastHistory ?? [];
    for (const record of pool) {
      const horizon = record?.horizon_hours;
      if (horizon === null || horizon === undefined) continue;
      const key = String(horizon);
      if (!out[key]) out[key] = {};
      for (const variable of AUDITED_VARIABLES) {
        const producer = record.variables?.[variable]?.producer ?? null;
        const existing = out[key][variable];
        // Only record agreement; a value that varies across records is itself a
        // finding, and leaving the field null surfaces it as "unknown" rather
        // than picking whichever record happened to be last.
        if (existing === undefined) out[key][variable] = producer;
        else if (existing !== producer) out[key][variable] = null;
      }
    }
    return out;
  }, [payload]);

  const matrix = useMemo(
    () => buildSourceMatrix({ policy: payload?.policy ?? null, observedProducers }),
    [observedProducers, payload]
  );

  const carryForward = useMemo(
    () => assessCarryForward(forecast, payload?.originObservation ?? null),
    [forecast, payload]
  );

  const verified = useMemo(
    () => buildVerifiedAccuracyTable({ verification: payload?.verification ?? null }),
    [payload]
  );

  const expectedSection = useMemo(
    () => buildExpectedAccuracySection(payload?.expectedAccuracy ?? null),
    [payload]
  );
  const expected = useMemo(() => buildExpectedAccuracyTable(expectedSection), [expectedSection]);

  const activeHorizon = forecast?.horizon_hours ?? selectedHorizon ?? null;

  /* ── render ──────────────────────────────────────────────────────────── */

  const outer = `rounded-lg border border-slate-800 bg-slate-900/40 font-mono text-slate-300 ${className}`;

  if (status === "error" && error) {
    return (
      <section className={outer} data-testid="prediction-provenance-panel" data-state="error">
        <div className="flex items-start gap-2 p-3">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-rose-400" aria-hidden />
          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-semibold text-rose-300">
              Provenance unavailable ({error.kind}
              {error.status ? ` ${error.status}` : ""})
            </p>
            <p className="mt-0.5 text-[11px] leading-snug text-slate-400">{error.message}</p>
            <button
              type="button"
              onClick={() => setReloadToken((n) => n + 1)}
              className="mt-2 inline-flex items-center gap-1 border border-slate-700 px-2 py-1 text-[10px] uppercase tracking-wide text-slate-300 hover:bg-slate-800"
            >
              <RefreshCw className="h-3 w-3" aria-hidden />
              retry
            </button>
          </div>
        </div>
      </section>
    );
  }

  if (status === "loading" || !payload) {
    return (
      <section className={outer} data-testid="prediction-provenance-panel" data-state="loading">
        <div className="flex items-center gap-2 p-3 text-[11px] text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          reading bounded audit tail for {stationId}…
        </div>
      </section>
    );
  }

  const provenance = forecast?.provenance ?? null;
  const model = provenance?.model ?? null;
  const policy = provenance?.policy ?? null;
  const nwp = provenance?.nwp ?? null;
  const bundleKey = activeHorizon !== null ? `h${activeHorizon}` : null;
  const bundle = bundleKey ? payload.bundles?.[bundleKey] ?? null : null;
  const originAge = minutesBetween(forecast?.recorded_at_utc ?? null, forecast?.origin_timestamp_utc ?? null);
  const target =
    forecast?.origin_timestamp_utc && activeHorizon !== null
      ? new Date(
          Date.parse(forecast.origin_timestamp_utc) + activeHorizon * 3600_000
        ).toISOString()
      : null;
  const warnings = payload.warnings ?? [];

  return (
    <section
      className={outer}
      data-testid="prediction-provenance-panel"
      data-state="ready"
      data-station-id={stationId}
      data-record-id={forecast?.record_id ?? ""}
      data-horizon-hours={activeHorizon ?? ""}
      data-persistence-cells={matrix.persistenceCells}
      data-total-cells={matrix.totalCells}
      aria-label="Forecast provenance"
    >
      {/* ── identity bar ──────────────────────────────────────────────── */}
      <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-slate-800 bg-slate-950/60 px-3 py-1.5">
        <Database className="h-3.5 w-3.5 shrink-0 text-slate-500" aria-hidden />
        <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-300">
          provenance
        </span>
        <span className="text-[11px] text-slate-400" data-testid="provenance-station">
          {stationName ? `${stationName} · ` : ""}
          {stationId}
        </span>
        <span className="text-[10px] text-slate-600">record</span>
        <span className="font-mono text-[10px] text-slate-400" data-testid="provenance-record-id">
          {forecast?.record_id ?? "—"}
        </span>
        <span className="ml-auto flex items-center gap-2">
          {refreshing ? (
            <Loader2 className="h-3 w-3 animate-spin text-slate-600" aria-label="Refreshing" />
          ) : (
            <button
              type="button"
              onClick={() => setReloadToken((n) => n + 1)}
              className="text-[10px] uppercase tracking-wide text-slate-500 hover:text-slate-300"
            >
              refresh
            </button>
          )}
          <span className="text-[10px] text-slate-600">read {formatUtc(payload.generatedAtUtc)}</span>
        </span>
      </header>

      <div className="space-y-3 p-3">
        {/* ── 0. recent records for this station ─────────────────────── */}
        {payload.forecastHistory && payload.forecastHistory.length > 1 ? (
          <section aria-label="Recent published records">
            <SectionLabel
              right={
                <span>
                  {payload.forecastHistory.length} record
                  {payload.forecastHistory.length === 1 ? "" : "s"} in the read window
                </span>
              }
            >
              record · origin → published
            </SectionLabel>
            <div className="mt-1.5 flex flex-wrap gap-1">
              {[...payload.forecastHistory]
                .reverse()
                .slice(0, 12)
                .map((record) => {
                  const isActive = record.record_id === forecast?.record_id;
                  const status = record.extra?.prediction_status;
                  const isFailure = typeof status === "string" && status !== "READY";
                  return (
                    <button
                      key={record.record_id}
                      type="button"
                      onClick={() => setSelectedRecordId(record.record_id)}
                      aria-pressed={isActive}
                      title={`${record.record_id}\norigin ${formatUtc(record.origin_timestamp_utc)}\nwritten ${formatUtc(record.recorded_at_utc)}${isFailure ? `\nstatus ${status}` : ""}`}
                      className={`border px-1.5 py-0.5 text-left font-mono text-[10px] tabular-nums ${
                        isActive
                          ? "border-slate-500 bg-slate-800/70 text-slate-100"
                          : isFailure
                            ? "border-rose-900/70 text-rose-300 hover:bg-slate-800/50"
                            : "border-slate-800 text-slate-500 hover:bg-slate-800/50 hover:text-slate-300"
                      }`}
                    >
                      +{String(record.horizon_hours ?? "?")}h
                      <span className="ml-1 text-slate-600">
                        {formatUtc(record.origin_timestamp_utc).slice(5, 16)}
                      </span>
                      {isFailure ? <span className="ml-1 text-rose-400">!</span> : null}
                    </button>
                  );
                })}
              {selectedRecordId ? (
                <button
                  type="button"
                  onClick={() => setSelectedRecordId(null)}
                  className="border border-slate-800 px-1.5 py-0.5 text-[10px] text-slate-500 hover:text-slate-300"
                >
                  newest
                </button>
              ) : null}
            </div>
          </section>
        ) : null}

        {/* ── 1. artifact identity ───────────────────────────────────── */}
        <section aria-label="Model artifacts">
          <SectionLabel
            right={
              <span className="flex flex-wrap justify-end gap-x-3 gap-y-0.5">
                <span>policy {policy?.policy_version ?? "—"}</span>
                <span>commit {shortSha(policy?.policy_code_commit)}</span>
              </span>
            }
          >
            model · checkpoint · policy
          </SectionLabel>
          <div className="mt-1.5 grid grid-cols-2 gap-1 sm:grid-cols-3 lg:grid-cols-5">
            <Cell title={`bundle for the +${activeHorizon ?? "?"}h horizon`}>
              <span className="text-slate-500">bundle </span>
              {model?.bundle ?? bundleKey ?? "—"}
            </Cell>
            <Cell
              title={
                bundle?.checkpointSha256
                  ? bundle.checkpointSha256
                  : "checkpoint sha256 from the bundle manifest"
              }
            >
              <span className="text-slate-500">ckpt </span>
              {shortSha(bundle?.checkpointSha256 ?? model?.checkpoint_sha256)}
            </Cell>
            <Cell title="when the checkpoint was trained">
              <span className="text-slate-500">trained </span>
              {formatUtc(bundle?.trainingTimestamp ?? model?.checkpoint_trained_at)}
            </Cell>
            <Cell title="bundle model family and seed">
              <span className="text-slate-500">family </span>
              {bundle?.modelFamily ?? "—"}
              {bundle?.randomSeed !== null && bundle?.randomSeed !== undefined
                ? ` s${bundle.randomSeed}`
                : ""}
            </Cell>
            <Cell title="conformal uncertainty reported by the bundle">
              <span className="text-slate-500">uncert </span>
              {bundle?.uncertaintyStatus ?? "—"}
            </Cell>
          </div>

          <div className="mt-1.5 grid grid-cols-2 gap-1 sm:grid-cols-4">
            <Cell title="forecast origin: the last observation that seeded this run">
              <span className="text-slate-500">origin </span>
              {formatUtc(forecast?.origin_timestamp_utc)}
            </Cell>
            <Cell title="target time this forecast is predicting">
              <span className="text-slate-500">target </span>
              {formatUtc(target)}
            </Cell>
            <Cell title="when the audit record was written">
              <span className="text-slate-500">written </span>
              {formatUtc(forecast?.recorded_at_utc)}
            </Cell>
            <Cell title="gap between the origin observation and the published record">
              <span className="text-slate-500">lag </span>
              {formatMinutes(originAge)}
            </Cell>
          </div>

          <div className="mt-1.5 grid grid-cols-1 gap-1 sm:grid-cols-2">
            <Cell
              className={nwp?.applied ? "border-sky-800/70" : "border-slate-800/70"}
              title="NWP router: applied, or why it was not used"
            >
              <span className="text-slate-500">nwp </span>
              {nwp?.applied ? "applied" : "not used"}
              {nwp?.reason ? <span className="text-slate-500"> · {nwp.reason}</span> : null}
            </Cell>
            <Cell title="wind sensor verdict carried on the same record as this forecast">
              <span className="text-slate-500">sensor </span>
              {forecast?.sensor_health?.verdict ?? "—"}
              {forecast?.sensor_health?.reason ? (
                <span className="text-slate-500"> · {forecast.sensor_health.reason}</span>
              ) : null}
              {forecast?.sensor_health?.n ? (
                <span className="text-slate-600"> (n={forecast.sensor_health.n})</span>
              ) : null}
            </Cell>
          </div>
        </section>

        {/* ── 2. the 20-cell source matrix ───────────────────────────── */}
        <section aria-label="Source selection across all horizons">
          <SectionLabel
            right={
              <>
                <span className="text-amber-300">{matrix.persistenceCells} persistence</span>{" "}
                <span className="text-slate-500">/</span>{" "}
                <span className="text-slate-400">{matrix.totalCells} cells</span>
                {matrix.unknownCells > 0 ? (
                  <>
                    {" · "}
                    <span className="text-slate-500">{matrix.unknownCells} unknown</span>
                  </>
                ) : null}
              </>
            }
          >
            model vs persistence · every horizon × variable
          </SectionLabel>

          <div className="mt-1.5 overflow-x-auto">
            <table className="w-full border-collapse text-[11px]">
              <thead>
                <tr>
                  <th className="w-24 px-1 py-0.5 text-left text-[10px] font-normal uppercase tracking-wide text-slate-500">
                    variable
                  </th>
                  {matrix.horizons.map((h) => (
                    <th key={h} className="p-0 text-center">
                      {onSelectHorizon ? (
                        <button
                          type="button"
                          onClick={() => handleHorizon(h)}
                          aria-pressed={h === activeHorizon}
                          className={`w-full px-1 py-0.5 text-[10px] uppercase tracking-wide hover:text-slate-200 ${
                            h === activeHorizon
                              ? "bg-slate-800/70 text-slate-100 underline decoration-slate-500 underline-offset-2"
                              : "text-slate-500"
                          }`}
                        >
                          +{h}h
                        </button>
                      ) : (
                        <span
                          className={`block px-1 py-0.5 text-[10px] uppercase tracking-wide ${
                            h === activeHorizon ? "text-slate-200" : "text-slate-500"
                          }`}
                        >
                          +{h}h
                        </span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {AUDITED_VARIABLES.map((variable) => (
                  <tr key={variable} className="border-t border-slate-800/60">
                    <th
                      scope="row"
                      className="px-1 py-0.5 text-left text-[10px] font-normal text-slate-400"
                    >
                      {VARIABLE_UNITS[variable].label}
                    </th>
                    {matrix.horizons.map((h) => {
                      const cell = matrix.cells.find(
                        (c) => c.horizonHours === h && c.variable === variable
                      );
                      if (!cell) {
                        return (
                          <td key={h} className="px-1 py-0.5 text-center">
                            <Gap>no policy entry</Gap>
                          </td>
                        );
                      }
                      return (
                        <td
                          key={h}
                          className={`px-1 py-0.5 text-center ${
                            h === activeHorizon ? "bg-slate-800/50" : ""
                          }`}
                          title={
                            cell.conflictReason ??
                            `policy source: ${cell.policySource ?? "unknown"}; record producer: ${
                              cell.recordedProducer ?? "none"
                            }`
                          }
                        >
                          <div className="flex flex-col items-center gap-0.5">
                            <ServedByTag servedBy={cell.servedBy} />
                            {cell.labelConflict ? (
                              <ConflictMark reason={cell.conflictReason} />
                            ) : null}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-1 text-[10px] leading-snug text-slate-500">
            {matrix.persistenceCells} of {matrix.totalCells} horizon × variable cells are served from the
            origin observation. {matrix.modelCells} come from a model.
            {matrix.conflictCells > 0
              ? ` ${matrix.conflictCells} cells have a recorded producer that disagrees with the policy.`
              : ""}
          </p>

          {matrix.producerLabelWarning ? (
            <div
              className="mt-1.5 flex items-start gap-2 border border-rose-900/70 bg-rose-950/40 px-2 py-1.5"
              data-testid="producer-label-warning"
            >
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-rose-400" aria-hidden />
              <p className="text-[10px] leading-snug text-rose-200">{matrix.producerLabelWarning}</p>
            </div>
          ) : null}
        </section>

        {/* ── 3. this cell, in detail ────────────────────────────────── */}
        <section aria-label={`Published values at +${activeHorizon ?? "?"}h`}>
          <SectionLabel
            right={
              payload.originObservation ? (
                <span title="The observation a persistence value is carried forward from">
                  obs {formatUtc(payload.originObservation.observedAtUtc)} · age{" "}
                  {formatMinutes(payload.originObservation.ageMinutes)}
                </span>
              ) : (
                <span>origin observation not available</span>
              )
            }
          >
            published values · +{activeHorizon ?? "?"}h
          </SectionLabel>

          {!forecast ? (
            <p className="mt-1.5 text-[11px] text-slate-500">
              No audit record found for this station and horizon. The forecast may predate the trail, or the
              tail read did not reach it.
            </p>
          ) : (
            <div className="mt-1.5 overflow-x-auto">
              <table className="w-full border-collapse text-[11px]">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wide text-slate-500">
                    <th className="px-1 py-0.5 text-left font-normal">variable</th>
                    <th className="px-1 py-0.5 text-right font-normal">published</th>
                    <th className="px-1 py-0.5 text-center font-normal">served by</th>
                    <th className="px-1 py-0.5 text-left font-normal">policy source</th>
                    <th className="px-1 py-0.5 text-left font-normal">record says</th>
                    <th className="px-1 py-0.5 text-left font-normal">vs origin obs</th>
                    <th className="px-1 py-0.5 text-right font-normal">expected error</th>
                  </tr>
                </thead>
                <tbody>
                  {AUDITED_VARIABLES.map((variable) => {
                    const entry = forecast.variables?.[variable] ?? null;
                    const cell = matrix.cells.find(
                      (c) => c.horizonHours === activeHorizon && c.variable === variable
                    );
                    const carry = carryForward.find((c) => c.variable === variable) ?? null;
                    const expectedPct = skillOverPersistence(expectedSection, variable, activeHorizon ?? -1);
                    const expectedCell = expectedSection.variables?.[variable]?.find(
                      (c) => c.horizonHours === activeHorizon
                    );
                    const expectedValue =
                      expectedCell?.status === "MEASURED" && typeof expectedCell.value === "number"
                        ? expectedCell.value
                        : null;

                    return (
                      <tr key={variable} className="border-t border-slate-800/60 align-top">
                        <th scope="row" className="px-1 py-1 text-left text-[10px] font-normal text-slate-400">
                          {VARIABLE_UNITS[variable].label}
                        </th>
                        <td className="px-1 py-1 text-right tabular-nums text-slate-200">
                          {formatNumber(entry?.value)} {VARIABLE_UNITS[variable].auditUnit}
                        </td>
                        <td className="px-1 py-1 text-center">
                          {cell ? <ServedByTag servedBy={cell.servedBy} /> : <Gap>no cell</Gap>}
                        </td>
                        <td className="px-1 py-1 text-[10px] text-slate-400">
                          {cell?.policySource ?? <Gap>unknown</Gap>}
                        </td>
                        <td className="px-1 py-1">
                          <ConflictMark reason={cell?.conflictReason ?? null} />
                        </td>
                        <td className="px-1 py-1">
                          <CarryForwardCell carry={carry} />
                        </td>
                        <td className="px-1 py-1 text-right tabular-nums">
                          {expectedValue !== null ? (
                            <span
                              className="text-slate-300"
                              title={`Offline holdout ${VARIABLE_UNITS[variable].metric === "brier" ? "Brier" : "MAE"}; not verified against live observations`}
                            >
                              ±{formatMetric(expectedValue)}
                              <span className="text-slate-600">
                                {" "}
                                {VARIABLE_UNITS[variable].canonicalUnit}
                              </span>
                              {expectedPct !== null ? (
                                <span className="text-emerald-400"> ↓{expectedPct.toFixed(1)}% vs persist</span>
                              ) : null}
                            </span>
                          ) : (
                            <ExpectedGap
                              status={(expectedCell?.status ?? "NOT_MEASURED") as ExpectedCellStatus}
                              verifiedExists={verified.exists}
                              pending={verified.pending}
                            />
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* ── 4. verified accuracy, per variable only ────────────────── */}
        <section aria-label="Verified accuracy">
          <SectionLabel
            right={
              <span className={verified.hasNoVerifiedHistory ? "text-amber-300" : "text-slate-400"}>
                {verified.hasNoVerifiedHistory
                  ? `no verified history · ${verified.pending} pending`
                  : `${verified.scored} scored`}
              </span>
            }
          >
            verified accuracy · scored against observations
          </SectionLabel>

          <div className="mt-1.5 overflow-x-auto">
            <table className="w-full border-collapse text-[11px]">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide text-slate-500">
                  <th className="px-1 py-0.5 text-left font-normal">variable</th>
                  <th className="px-1 py-0.5 text-right font-normal">mae</th>
                  <th className="px-1 py-0.5 text-right font-normal">bias</th>
                  <th className="px-1 py-0.5 text-right font-normal">n</th>
                  <th className="px-1 py-0.5 text-left font-normal">by producer</th>
                </tr>
              </thead>
              <tbody>
                {verified.variables.map((row) => {
                  const note = conversionNote(row.variable, row.auditUnit);
                  return (
                    <tr
                      key={row.variable}
                      className="border-t border-slate-800/60 align-top"
                      data-testid={`verified-row-${row.variable}`}
                    >
                      <th scope="row" className="px-1 py-1 text-left text-[10px] font-normal text-slate-400">
                        {row.label}
                      </th>
                      <td className="px-1 py-1 text-right tabular-nums text-slate-200">
                        {row.reportedMae !== null || row.sampledMae !== null ? (
                          <>
                            {formatMetric(row.reportedMae ?? row.sampledMae)}{" "}
                            <span className="text-slate-600">{row.canonicalUnit}</span>
                            {note ? <div className="text-[9px] text-slate-600">{note}</div> : null}
                          </>
                        ) : (
                          <Gap>no verified score</Gap>
                        )}
                      </td>
                      <td
                        className="px-1 py-1 text-right tabular-nums text-slate-300"
                        title="Mean signed error over the read window: positive means the model ran hot."
                      >
                        {row.sampledMeanSignedError !== null ? (
                          formatMetric(row.sampledMeanSignedError, 3)
                        ) : (
                          <Gap title="Not computable without scored samples.">—</Gap>
                        )}
                      </td>
                      <td
                        className="px-1 py-1 text-right tabular-nums text-slate-400"
                        title={
                          row.sampledN > 0
                            ? `${row.sampledN} scored sample(s) in the bounded read window`
                            : "No scored samples. Not a measured error of zero."
                        }
                      >
                        {row.sampledN > 0 ? row.sampledN : <Gap title="no scored samples">none</Gap>}
                      </td>
                      <td className="px-1 py-1 text-[10px] text-slate-400">
                        {row.byProducer.length > 0 ? (
                          row.byProducer.map((p) => (
                            <div key={p.producer} className="tabular-nums">
                              {p.producer}{" "}
                              <span className="text-slate-600">
                                mae {formatMetric(p.meanAbsError)} (n={p.n})
                              </span>
                            </div>
                          ))
                        ) : (
                          <Gap>{row.emptyReason ?? "no producer breakdown"}</Gap>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="mt-1 text-[10px] leading-snug text-slate-500">{verified.summary}</p>
          <p className="mt-0.5 text-[10px] leading-snug text-slate-600">
            Errors are never averaged across variables: degC, %RH, hPa and m/s do not share a scale, so a
            single pooled figure would be arithmetically valid and scientifically meaningless.
            {verified.sourcePath ? ` Source: ${verified.sourcePath}.` : ""}
            {verified.toleranceMinutes !== null
              ? ` Observation tolerance ±${verified.toleranceMinutes} min.`
              : ""}
          </p>
        </section>

        {/* ── 5. expected accuracy, static ────────────────────────────── */}
        <section aria-label="Expected accuracy from the offline benchmark">
          <SectionLabel
            right={
              <span title={expectedSection.source}>
                {expectedSection.asOf ?? "—"} · static reference
              </span>
            }
          >
            expected accuracy · offline benchmark, not verified
          </SectionLabel>

          <div className="mt-1.5 overflow-x-auto">
            <table className="w-full border-collapse text-[11px]">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide text-slate-500">
                  <th className="w-24 px-1 py-0.5 text-left font-normal">variable</th>
                  {matrix.horizons.map((h) => (
                    <th
                      key={h}
                      className={`px-1 py-0.5 text-right font-normal ${
                        h === activeHorizon ? "text-slate-200" : "text-slate-500"
                      }`}
                    >
                      +{h}h
                    </th>
                  ))}
                  {expected.some((r) => r.outsideAuditedCells) ? (
                    <th className="px-1 py-0.5 text-right font-normal text-slate-600">rain</th>
                  ) : null}
                </tr>
              </thead>
              <tbody>
                {expected.map((row) => (
                  <tr
                    key={row.variable}
                    className={`border-t border-slate-800/60 ${row.outsideAuditedCells ? "opacity-80" : ""}`}
                  >
                    <th
                      scope="row"
                      className="px-1 py-1 text-left text-[10px] font-normal text-slate-400"
                      title={
                        row.outsideAuditedCells
                          ? "Rain occurrence is carried in extra.chance_of_rain_pct, not in variables[]. Scored with Brier, which is unitless."
                          : row.canonicalUnit
                      }
                    >
                      {row.label}
                      {row.outsideAuditedCells ? (
                        <span className="ml-1 text-slate-600">(brier)</span>
                      ) : null}
                    </th>
                    {matrix.horizons.map((h) => {
                      const cell = row.cells.find((c) => c.horizonHours === h);
                      if (!cell) {
                        return (
                          <td key={h} className="px-1 py-1 text-right">
                            <Gap>—</Gap>
                          </td>
                        );
                      }
                      if (cell.status === "MEASURED" && typeof cell.value === "number") {
                        return (
                          <td
                            key={h}
                            className={`px-1 py-1 text-right tabular-nums ${
                              h === activeHorizon ? "text-slate-100" : "text-slate-300"
                            }`}
                            title={`${cell.metric === "brier" ? "Brier score" : "MAE"}; beats ${cell.beatsBaselines.join(", ") || "nothing recorded"}`}
                          >
                            {formatMetric(cell.value)}
                            {cell.beatsPersistencePct !== null ? (
                              <span className="ml-1 text-[9px] text-emerald-400">
                                ↓{cell.beatsPersistencePct.toFixed(1)}%
                              </span>
                            ) : null}
                          </td>
                        );
                      }
                      return (
                        <td key={h} className="px-1 py-1 text-right">
                          <Gap title={describeCellGap(cell.status, {
                            verifiedExists: verified.exists,
                            pending: verified.pending,
                          })}>
                            {cell.status === "PERSISTENCE_EQUIVALENT" ? "= persist" : "not measured"}
                          </Gap>
                        </td>
                      );
                    })}
                    {expected.some((r) => r.outsideAuditedCells) ? (
                      <td className="px-1 py-1 text-right text-[10px] text-slate-600">
                        {row.outsideAuditedCells ? (
                          <span className="tabular-nums">
                            {row.cells
                              .filter((c) => c.status === "MEASURED")
                              .map((c) => `+${c.horizonHours}h ${formatMetric(c.value)}`)
                              .join(" · ") || "—"}
                          </span>
                        ) : null}
                      </td>
                    ) : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-1 text-[10px] leading-snug text-slate-600">
            {expectedSection.coverageNote ??
              "Static offline benchmark. Not measured against live traffic."}
          </p>
        </section>

        {/* ── 6. read coverage ───────────────────────────────────────── */}
        <section aria-label="Read coverage">
          <SectionLabel>read coverage</SectionLabel>
          <div className="mt-1.5 grid grid-cols-1 gap-1 sm:grid-cols-2">
            <Cell title={describeCoverage(payload.forecastCoverage)}>
              <span className="text-slate-500">trail </span>
              {describeCoverage(payload.forecastCoverage)}
            </Cell>
            <Cell title={describeCoverage(payload.verification?.coverage ?? null)}>
              <span className="text-slate-500">verify </span>
              {payload.verification?.exists
                ? describeCoverage(payload.verification?.coverage ?? null)
                : "verification trail does not exist"}
            </Cell>
          </div>
          {warnings.length > 0 ? (
            <ul className="mt-1.5 space-y-0.5" data-testid="provenance-warnings">
              {warnings.map((warning, index) => (
                <li key={index} className="font-mono text-[10px] text-amber-300">
                  ▲ {warning}
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      </div>
    </section>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
 * Cell renderers that carry the "was this just the last observation?" question
 * ──────────────────────────────────────────────────────────────────────────── */

function CarryForwardCell({ carry }: { carry: CarryForwardVerdict | null }) {
  if (!carry) return <Gap>—</Gap>;
  if (carry.publishedValue === null) return <Gap>no value</Gap>;
  if (carry.observationValue === null) {
    return <Gap title="The origin observation was outside the bounded read window">origin obs not read</Gap>;
  }
  if (carry.carriedForward) {
    return (
      <span
        className="inline-block border border-amber-700/70 bg-amber-950/50 px-1 font-mono text-[10px] text-amber-300"
        title={`Identical to the origin observation (${carry.observationValue} ${carry.auditUnit}). Carried forward, not predicted.`}
      >
        carried forward
      </span>
    );
  }
  return (
    <span
      className="font-mono text-[10px] tabular-nums text-slate-400"
      title={`Published minus origin observation, in ${carry.auditUnit}. Non-zero, so the value is not the observation itself.`}
    >
      {carry.delta! > 0 ? "+" : ""}
      {formatNumber(carry.delta, 3)} {carry.auditUnit}
    </span>
  );
}

function ExpectedGap({
  status,
  verifiedExists,
  pending,
}: {
  status: ExpectedCellStatus;
  verifiedExists: boolean;
  pending: number;
}) {
  const text = describeCellGap(status, { verifiedExists, pending });
  return <Gap title={text}>{status === "PERSISTENCE_EQUIVALENT" ? "= persist" : "not measured"}</Gap>;
}
