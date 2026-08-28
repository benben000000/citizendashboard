"use client";

import { CloudSun, Waves, Search } from "lucide-react";
import { useTranslations } from "next-intl";

interface DashboardLoadingScreenProps {
  message?: string;
  destination?: "weather" | "water-level" | "prediction" | null;
}

export default function DashboardLoadingScreen({
  message,
  destination,
}: DashboardLoadingScreenProps) {
  const t = useTranslations("routeTransition");
  const Icon =
    destination === "water-level"
      ? Waves
      : destination === "prediction"
        ? Search
        : CloudSun;

  const label =
    message ??
    (destination === "water-level"
      ? t("waterLevel")
      : destination === "prediction"
        ? (t.has("prediction") ? t("prediction") : "Loading prediction")
        : destination === "weather"
          ? t("weather")
          : t("fallback"));

  return (
    <div
      className="fixed inset-0 z-50 flex min-h-svh items-center justify-center bg-white/60 px-6 backdrop-blur-md"
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-col items-center text-center">
        <div className="relative h-14 w-14">
          <div className="absolute inset-0 animate-spin rounded-full border-4 border-slate-950/10 border-t-main" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Icon className="h-6 w-6 text-slate-800" aria-hidden="true" strokeWidth={2.25} />
          </div>
        </div>
        <p className="mt-4 text-sm font-semibold uppercase tracking-wide text-slate-800">
          {label}
        </p>
        <span className="sr-only">{t("srOnly")}</span>
      </div>
    </div>
  );
}
