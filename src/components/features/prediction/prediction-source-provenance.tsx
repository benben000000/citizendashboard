"use client";

/**
 * PredictionSourceProvenance
 *
 * A compact, per-variable "where did this number come from?" strip.
 *
 * WHY THIS IS A STRIP AND NOT A BANNER
 * ------------------------------------
 * A governance banner was deliberately removed from this surface: it dominated
 * the viewport above the forecast. This is deliberately not a replacement. It is
 * one muted line of six chips, sitting under the forecast, sized to be read in
 * passing by an operator and expanded only on demand.
 *
 * WHAT IT IS FOR
 * --------------
 * The served policy (`prediction-model/data/inference_policy.json`, v2.0.0)
 * routes most (horizon, variable) cells to `persistence_fallback`. At 1h and 3h
 * that is ALL five policy variables: the numbers on screen are the last
 * observation, carried forward. Rendered without a source label they read as a
 * forecast, and an operator has no way to tell the difference. The chips make
 * that difference legible in a glance, per variable, at the horizon in view.
 *
 * The raw policy token is shown verbatim rather than prettified, because the
 * token is what an operator will grep for in `inference_policy.json`. Prose in
 * parentheses carries the plain-English meaning.
 *
 * NO PROSE CLAIMS OF SKILL
 * ------------------------
 * Nothing here says a value is "accurate" or "unreliable". This component
 * reports provenance only. Measured error lives in the disclosure below it and
 * comes exclusively from `verify_predictions.py`; when it has produced nothing,
 * that is what is said.
 *
 * STRINGS
 * -------
 * The i18n catalogues (`src/lib/i18n/messages/*.ts`) are outside this change's
 * blast radius, so these strings are inline English. That is the same choice
 * `model-governance-banner.tsx` already makes, and it is also the right one for
 * a technical token — the token is the fact, the gloss is the help text.
 */

import { useMemo, type ComponentType } from "react";
import {
  Cpu,
  History,
  Sigma,
  ShieldAlert,
  CircleHelp,
  type LucideProps,
} from "lucide-react";

import type { ModelSourceSelection } from "@/types/prediction";
import {
  buildSourceProvenance,
  verifiedAccuracyUnavailable,
  type ForecastSourceId,
  type ForecastSourceProvenance,
  type ForecastVariable,
  type ForecastVariableSource,
  type VerifiedAccuracy,
} from "@/types/forecast-provenance";

interface PredictionSourceProvenanceProps {
  /**
   * Raw per-variable source selection from the response. Absent means no model
   * forecast backs this response — the component then says so instead of
   * rendering empty chips.
   */
  sourceSelection?: ModelSourceSelection;
  /** The horizon in view, so the strip is never ambiguous about what it covers. */
  horizon: string;
  /** Live-measured accuracy, when the service has attached it. */
  verifiedAccuracy?: VerifiedAccuracy;
  className?: string;
}

const VARIABLE_SHORT_LABEL: Readonly<Record<ForecastVariable, string>> = Object.freeze({
  temperature: "Temp",
  humidity: "Hum",
  pressure: "Pres",
  windSpeed: "Wind",
  windDirection: "Dir",
  heatIndex: "HI",
});

const VARIABLE_LONG_LABEL: Readonly<Record<ForecastVariable, string>> = Object.freeze({
  temperature: "Temperature",
  humidity: "Humidity",
  pressure: "Pressure",
  windSpeed: "Wind speed",
  windDirection: "Wind direction",
  heatIndex: "Heat index",
});

interface SourcePresentation {
  /** The policy token, shown verbatim. This is the auditable fact. */
  token: string;
  icon: ComponentType<LucideProps>;
  /** Chip tint. Model output is the only one that gets the "live" colour. */
  chipClass: string;
  dotClass: string;
  /** Plain-English meaning, shown in the tooltip. */
  gloss: string;
}

/**
 * Presentation per source. `unknown` is styled as absent evidence, never as a
 * soft variant of "model" — an unrecognised token must not read as skill.
 */
const SOURCE_PRESENTATION: Readonly<Record<ForecastSourceId, SourcePresentation>> = Object.freeze({
  learned_model: {
    token: "learned_model",
    icon: Cpu,
    chipClass: "border-emerald-400/50 bg-emerald-400/10 text-emerald-200",
    dotClass: "bg-emerald-400",
    gloss: "Produced by the neural network for this horizon.",
  },
  persistence_fallback: {
    token: "persistence_fallback",
    icon: History,
    chipClass: "border-slate-400/40 bg-slate-400/10 text-slate-200",
    dotClass: "bg-slate-400",
    gloss: "The last observation, carried forward unchanged. Not a forecast.",
  },
  derived: {
    token: "derived",
    icon: Sigma,
    chipClass: "border-sky-400/40 bg-sky-400/10 text-sky-200",
    dotClass: "bg-sky-400",
    gloss: "Computed from the selected temperature and humidity. Not an independent forecast.",
  },
  suppressed: {
    token: "suppressed",
    icon: ShieldAlert,
    chipClass: "border-amber-400/50 bg-amber-400/10 text-amber-200",
    dotClass: "bg-amber-400",
    gloss: "A data-quality gate withheld the model output. The value is the last observation.",
  },
  unknown: {
    token: "unknown",
    icon: CircleHelp,
    chipClass: "border-slate-600/60 bg-slate-800/40 text-slate-400",
    dotClass: "bg-slate-600",
    gloss: "The engine emitted a source token this build does not recognise. Not assumed to be model output.",
  },
});

/** Build a renderable provenance block from the DTO's raw selection. */
export function provenanceFromSelection(
  sourceSelection: ModelSourceSelection | undefined,
  horizon: string
): ForecastSourceProvenance | null {
  if (!sourceSelection) return null;
  return buildSourceProvenance({
    horizon,
    selectedSourceByVariable: {
      temperature: sourceSelection.temperature,
      humidity: sourceSelection.humidity,
      pressure: sourceSelection.pressure,
      wind_speed: sourceSelection.windSpeed,
      wind_direction: sourceSelection.windDirection,
      heat_index: sourceSelection.heatIndex,
    },
  });
}

function formatAge(seconds: number | null): string | null {
  if (seconds === null) return null;
  if (seconds < 90) return `${seconds}s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 172800) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

export default function PredictionSourceProvenance({
  sourceSelection,
  horizon,
  verifiedAccuracy,
  className,
}: PredictionSourceProvenanceProps) {
  const provenance = useMemo(
    () => provenanceFromSelection(sourceSelection, horizon),
    [sourceSelection, horizon]
  );

  // Absent accuracy must read as absent. `verifiedAccuracyUnavailable` gives the
  // same explicit status the service would, so the UI never has to invent one.
  const accuracy: VerifiedAccuracy =
    verifiedAccuracy ??
    verifiedAccuracyUnavailable(
      "No verification summary is attached to this response, so no measured error is shown."
    );

  if (!provenance) {
    return (
      <div
        role="status"
        data-testid="source-provenance-unavailable"
        data-horizon={horizon}
        className={className}
      >
        <p className="text-[11px] font-medium uppercase tracking-wide text-amber-300/90">
          No model output at this horizon
        </p>
        <p className="mt-0.5 text-[11px] text-light/60">
          Every value below is a current observation, not a forecast.
        </p>
        <VerifiedAccuracyDisclosure accuracy={accuracy} />
      </div>
    );
  }

  const { modelOutputCount, carriedForwardCount, derivedCount, totalCount } = provenance;
  const unknownCount = provenance.unknownCount + provenance.suppressedCount;

  const summaryParts: string[] = [];
  summaryParts.push(
    modelOutputCount === 0
      ? `none of ${totalCount} are model output`
      : `${provenance.modelBackedLabel} are model output`
  );
  if (carriedForwardCount > 0) {
    summaryParts.push(
      `${carriedForwardCount} carried forward from the last observation`
    );
  }
  if (derivedCount > 0) summaryParts.push(`${derivedCount} derived`);
  if (unknownCount > 0) summaryParts.push(`${unknownCount} suppressed or unrecognised`);

  return (
    <div
      data-testid="source-provenance"
      data-horizon={provenance.horizon}
      data-model-output-count={modelOutputCount}
      data-carried-forward-count={carriedForwardCount}
      data-derived-count={derivedCount}
      data-unknown-count={unknownCount}
      data-total-count={totalCount}
      className={className}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-light/45">
          Source
        </span>

        {provenance.variables.map((v: ForecastVariableSource) => {
          const presentation = SOURCE_PRESENTATION[v.sourceId];
          const Icon = presentation.icon;
          return (
            <span
              key={v.variable}
              data-testid={`source-chip-${v.variable}`}
              data-source-id={v.sourceId}
              data-raw-token={v.rawToken ?? ""}
              title={`${VARIABLE_LONG_LABEL[v.variable]}: ${presentation.gloss} ` +
                `Policy token: ${v.rawToken ?? "not reported"}.`}
              className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium leading-none ${presentation.chipClass}`}
            >
              <Icon className="h-2.5 w-2.5 shrink-0" aria-hidden />
              <span className="tabular-nums">{VARIABLE_SHORT_LABEL[v.variable]}</span>
              <span
                aria-hidden
                className={`h-1 w-1 shrink-0 rounded-full ${presentation.dotClass}`}
              />
              <span className="sr-only">
                {VARIABLE_LONG_LABEL[v.variable]} source: {presentation.gloss}
              </span>
            </span>
          );
        })}
      </div>

      <p className="mt-1 text-[11px] leading-snug text-light/60">{summaryParts.join(" · ")}.</p>

      {provenance.rain.source ? (
        <p className="mt-0.5 text-[11px] leading-snug text-light/45">
          Rain probability is a blend ({provenance.rain.source}): model{" "}
          {provenance.rain.modelWeight ?? "—"} / persistence{" "}
          {provenance.rain.persistenceWeight ?? "—"}.
        </p>
      ) : null}

      <VerifiedAccuracyDisclosure accuracy={accuracy} />
    </div>
  );
}

/**
 * Measured error, or the reason there is none.
 *
 * Deliberately a `<details>`: it is reference material for an operator checking
 * whether to trust a number, not a headline. Expanded, it shows MAE per
 * variable AND per producer with its sample count — the sample count is not
 * decoration, a 0.4 degC MAE from n=3 is not evidence of anything.
 */
function VerifiedAccuracyDisclosure({ accuracy }: { accuracy: VerifiedAccuracy }) {
  const age = formatAge(accuracy.ageSeconds);
  const available = accuracy.status === "available";

  return (
    <details
      data-testid="verified-accuracy"
      data-accuracy-status={accuracy.status}
      className="mt-1.5"
    >
      <summary className="cursor-pointer select-none text-[11px] font-medium text-light/50 hover:text-light/80">
        {available
          ? `Measured error (${accuracy.entries.length} verified ${
              accuracy.entries.length === 1 ? "channel" : "channels"
            }${age ? `, ${age}` : ""})`
          : `Measured error: ${accuracy.status.replace(/_/g, " ")}`}
      </summary>

      <div className="mt-1.5 rounded-md border border-white/10 bg-black/20 p-2 text-[11px] text-light/70">
        {available ? (
          <>
            <table className="w-full text-left">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide text-light/40">
                  <th className="py-0.5 pr-2 font-medium">Variable</th>
                  <th className="py-0.5 pr-2 font-medium">Producer</th>
                  <th className="py-0.5 pr-2 text-right font-medium">MAE</th>
                  <th className="py-0.5 text-right font-medium">n</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {accuracy.entries.map((e) => (
                  <tr key={`${e.variable}:${e.producer}`} data-testid="verified-accuracy-row">
                    <td className="py-0.5 pr-2">{e.variable}</td>
                    <td className="py-0.5 pr-2">{e.producer}</td>
                    <td className="py-0.5 pr-2 text-right">
                      {e.mae.toFixed(3)}
                      {e.unit ? ` ${e.unit}` : ""}
                    </td>
                    <td className="py-0.5 text-right">{e.sampleCount}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-1.5 text-[10px] leading-snug text-light/45">{accuracy.note}</p>
          </>
        ) : (
          <p data-testid="verified-accuracy-unavailable" className="leading-snug">
            {accuracy.reason}
            {age ? ` Last pass: ${age}.` : null}
          </p>
        )}
      </div>
    </details>
  );
}
