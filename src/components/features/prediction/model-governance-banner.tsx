"use client";

import { useTranslations } from "next-intl";
import { AlertTriangle, Info, ShieldAlert, Clock } from "lucide-react";

import type {
  PredictionPublicDTO,
  ModelGovernance,
  ModelInputQuality,
  ModelSourceSelection,
} from "@/types/prediction";

/**
 * ModelGovernanceBanner
 *
 * Renders the forecast engine's OWN governance metadata on every prediction
 * surface.
 *
 * WHY THIS COMPONENT IS MANDATORY
 * ------------------------------
 * The engine emits `model_status: "RESEARCH_PROTOTYPE"` and
 * `not_for_life_safety: true` on every forecast. Those fields were previously
 * discarded at the service boundary, so the UI was free to render imperative
 * text such as "DANGER: FLOODED / DO NOT PASS" against a model that explicitly
 * declares it must not be used for life safety. Surfacing the metadata here
 * makes that class of failure visible instead of silent.
 *
 * Everything rendered is taken verbatim from the model response. Nothing here
 * is hardcoded status text that could drift from the engine.
 */
export default function ModelGovernanceBanner({ data }: { data: PredictionPublicDTO }) {
  const t = useTranslations("prediction.governance");

  const governance: ModelGovernance | undefined = data.governance;
  const quality: ModelInputQuality | undefined = data.inputQuality;
  const sources: ModelSourceSelection | undefined = data.sourceSelection;

  // --- No forecast available -------------------------------------------------
  if (!governance) {
    return (
      <div
        role="status"
        data-testid="forecast-unavailable"
        className="rounded-xl border border-amber-300/60 bg-amber-50/70 p-4 text-amber-900 dark:border-amber-500/40 dark:bg-amber-950/40 dark:text-amber-100"
      >
        <div className="flex items-start gap-3">
          <Info className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
          <div className="space-y-1">
            <p className="font-semibold">{t("noForecastTitle")}</p>
            <p className="text-sm">
              {data.forecastUnavailable?.message ??
                "No validated model forecast is currently available for this station."}
            </p>
            {data.forecastUnavailable?.reason ? (
              <p className="text-xs uppercase tracking-wide opacity-70">
                Reason: {data.forecastUnavailable.reason}
              </p>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  const quarantined = quality ? !quality.learnedOutputTrusted : false;
  const warning = quality?.sensorQualityFlag === "WARNING";

  const sourceLabel = (value: string | undefined): string => {
    if (!value) return t("sourcePersistence");
    if (value.includes("quarantined")) return t("sourceQuarantined");
    if (value.includes("learned") || value.includes("candidate")) return t("sourceLearned");
    return t("sourcePersistence");
  };

  return (
    <div className="space-y-3" data-testid="model-governance-banner">
      {/* Primary: research-prototype / not-for-life-safety notice */}
      <div
        role="note"
        data-model-status={governance.modelStatus}
        data-not-for-life-safety={String(governance.notForLifeSafety)}
        className="rounded-xl border border-amber-400/70 bg-amber-50/80 p-4 text-amber-950 dark:border-amber-500/50 dark:bg-amber-950/50 dark:text-amber-50"
      >
        <div className="flex items-start gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
          <div className="space-y-2">
            <p className="text-sm font-bold uppercase tracking-wide">
              {t("researchPrototypeBadge")}
            </p>
            <p className="text-sm leading-relaxed">{t("notForLifeSafetyNotice")}</p>
          </div>
        </div>
      </div>

      {/* Input quality gate */}
      {(quarantined || warning) && quality ? (
        <div
          role="alert"
          data-testid="input-quality-gate"
          className={`rounded-xl border p-4 text-sm ${
            quarantined
              ? "border-red-400/70 bg-red-50/80 text-red-950 dark:border-red-500/50 dark:bg-red-950/50 dark:text-red-50"
              : "border-yellow-400/70 bg-yellow-50/80 text-yellow-950 dark:border-yellow-500/50 dark:bg-yellow-950/50 dark:text-yellow-50"
          }`}
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
            <div className="space-y-1">
              <p className="font-semibold">
                {quarantined ? t("inputQuarantinedNotice") : t("inputWarningNotice")}
              </p>
              <p className="text-xs uppercase tracking-wide opacity-80">
                {quality.sensorQualityFlag} · {quality.anomalySummary}
              </p>
            </div>
          </div>
        </div>
      ) : null}

      {/* Provenance and freshness */}
      <details className="rounded-xl border border-border/70 bg-card/60 p-3 text-xs text-muted-foreground">
        <summary className="cursor-pointer select-none font-medium">
          {t("modelStatus", { status: governance.modelStatus })} ·{" "}
          {t("horizon", { horizon: governance.activeBundleHorizon })}
        </summary>
        <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
          <div>
            <dt className="inline font-medium">{t("modelVersion", {
              version: governance.modelVersion,
              policyVersion: governance.policyVersion,
            })}</dt>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" aria-hidden />
            <dt className="inline">
              {t("freshness", {
                age: governance.ageSeconds ?? "?",
                budget: governance.freshnessBudgetSeconds,
              })}
            </dt>
          </div>
          <div className="sm:col-span-2 break-all">
            <dt className="inline font-mono text-[11px]">
              {t("provenance", {
                commit: governance.modelCodeCommit.slice(0, 12),
                checkpoint: governance.checkpointSha256?.slice(0, 12) ?? "n/a",
              })}
            </dt>
          </div>
        </dl>

        {sources ? (
          <div className="mt-3 border-t border-border/60 pt-2">
            <p className="font-medium">Value sources (per-variable policy)</p>
            <ul className="mt-1 grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-3">
              {(
                [
                  ["Temperature", sources.temperature],
                  ["Humidity", sources.humidity],
                  ["Pressure", sources.pressure],
                  ["Wind speed", sources.windSpeed],
                  ["Wind direction", sources.windDirection],
                  ["Heat index", sources.heatIndex],
                ] as const
              ).map(([label, value]) => (
                <li key={label} className="flex justify-between gap-2">
                  <span>{label}</span>
                  <span
                    className={
                      value?.includes("learned") || value?.includes("candidate")
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-amber-600 dark:text-amber-400"
                    }
                  >
                    {sourceLabel(value)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {data.modelProvenance ? (
          <div className="mt-3 space-y-1 border-t border-border/60 pt-2">
            <p>
              Rain probability: {data.modelProvenance.rainProbabilitySource} (model{" "}
              {data.modelProvenance.rainModelWeight} / persistence{" "}
              {data.modelProvenance.rainPersistenceWeight})
            </p>
            {data.modelProvenance.weatherUncertaintyStatus === "UNAVAILABLE" ? (
              <p className="text-amber-600 dark:text-amber-400">
                {t("uncertaintyUnavailable")}
              </p>
            ) : null}
            {data.modelProvenance.blockedTargets.uvIndex.reason ? (
              <p>{t("uvBlocked", { reason: data.modelProvenance.blockedTargets.uvIndex.reason })}</p>
            ) : null}
            {data.modelProvenance.blockedTargets.lightIntensity.reason ? (
              <p>
                {t("lightBeta", {
                  reason: data.modelProvenance.blockedTargets.lightIntensity.reason,
                })}
              </p>
            ) : null}
            {data.modelProvenance.waterLevelBeta ? (
              <p className="font-medium text-amber-700 dark:text-amber-400">
                {t("waterBeta")}
              </p>
            ) : null}
          </div>
        ) : null}

        <p className="mt-2 text-[11px] italic opacity-70">{t("confidenceNote")}</p>
      </details>
    </div>
  );
}
