import { ElementType, ReactNode, useCallback, useMemo } from "react";
import { TelemetryMetrics, TelemetryPublicDTO } from "@/types/telemetry";
import { AlertTriangle, Sparkles } from "lucide-react";
import Link from "next/link";
import { useCardStyles, useMetricConfig } from "@/hooks/useWeatherMetric";
import { formatDate } from "@/lib/utils/date";
import { Skeleton } from "@/components/ui/skeleton";
import { useLocale, useTranslations } from "next-intl";
import type { Locale } from "@/lib/i18n/translations";

interface DataCardProps {
  metricKey: string;
  label: string;
  value: number | null | undefined;
  displayValue?: ReactNode;
  unit?: string;
  icon: ElementType;
  color: string;
  warning?: { color: string; term: string } | null;
  selectedMetric?: string | null;
  onMetricSelect?: (metricKey: string) => void;
}

interface StationMetricsCardProps {
  telemetryMetrics: TelemetryMetrics | null | undefined;
  selectedMetric?: string | null;
  onMetricSelect?: (metricKey: string) => void;
  stationData: TelemetryPublicDTO | null | undefined;
}

const DataCard = ({
  metricKey,
  label,
  value,
  displayValue,
  unit,
  icon: Icon,
  color,
  warning,
  selectedMetric,
  onMetricSelect,
}: DataCardProps) => {
  const isSelected = selectedMetric === metricKey;
  const isSelectable = !!onMetricSelect;
  const selectedBorderStyle = isSelected && isSelectable
    ? { borderLeft: "4px solid #E5E5E5 " }
    : {};
  const { hasWarning, warningStyles, cardStyle } = useCardStyles(warning, isSelected, isSelectable);

  const handleClick = useCallback(() => {
    if (isSelectable) {
      onMetricSelect(metricKey);
    }
  }, [metricKey, onMetricSelect, isSelectable]);

  const formattedValue = useMemo(() => {
    if (displayValue) return displayValue;

    if (value != null && typeof value === "number") {
      return value.toFixed(1);
    }
    //Fallback
    return "0";
  }, [value, displayValue]);

  return (
    <>
      {/* Desktop View */}
      <div
        className={`hidden glass min-h-1 md:flex md:h-full md:flex-col items-center justify-center md:min-h-35 p-5 md:p-4 transition-all duration-200 
          ${isSelectable ? "cursor-pointer hover:scale-105" : ""}
          
        `}
        style={{
          ...cardStyle,
          ...selectedBorderStyle,
          ...(isSelected
          ? {
              boxShadow: `
                inset 0 0.0625rem 0 rgba(255,255,255,0.4),
                0 0.625rem 1.875rem rgba(0,0,0,0.25),
                0 0 0 0.0625rem rgba(255,255,255,0.3)
              `,
            }
          : {}),
        }}
        onClick={handleClick}
      >
        {/* Top row: icon + label */}
        <div className="flex items-center gap-2 md:mb-2 md:w-full w-1/3">
          <div className="relative">
            <Icon className="h-4 w-4 md:h-5 md:w-5" color={"var(--light)"} />
            {hasWarning && (
              <AlertTriangle
                className="absolute -top-1.5 -right-1.5 w-3 h-3 animate-pulse"
                style={{ color: "var(--light)" }}
              />
            )}
          </div>
          <span className="text-xs md:text-sm text-light font-medium">{label}</span>
        </div>

        {/* Value — the most important element */}
        <div className="flex md:items-baseline gap-1 mt-2 md:w-full text-light w-1/3 justify-start items-center">
          <span className="text-xl md:text-2xl font-bold leading-none">{formattedValue}</span>
          {unit && <span className="text-sm  font-medium">{unit}</span>}
        </div>

        {/* Warning tag */}
        {hasWarning && warning?.term && (
            <div
              className="inline-flex items-center rounded-full py-0.5 md:text-base text-sm font-bold md:w-full w-1/3 justify-start "
            style={{
              backgroundColor: `${warningStyles.text}15`,
              color: "var(--light)",
            }}
          >
            {warning.term}
          </div>
        )}

        
      </div>

      {/* Mobile View */}
      <div
        className={`relative flex w-full flex-col p-4 rounded-xl border transition-all duration-200 md:hidden
          glass
          ${isSelectable ? "cursor-pointer hover:scale-105 active:scale-95" : ""}          
        `}
        style={{
          ...cardStyle,
          ...selectedBorderStyle,
          ...(isSelected
          ? {
              boxShadow: `
                inset 0 0.0625rem 0 rgba(255,255,255,0.4),
                0 0.625rem 1.875rem rgba(0,0,0,0.25),
                0 0 0 0.0625rem rgba(255,255,255,0.3)
              `,
            }
          : {}),
        }}
        onClick={handleClick}
      >
        {/* Label row */}
        <div className="flex items-center gap-1.5 mb-2">
          {hasWarning && (
            <AlertTriangle className="w-3 h-3 text-light animate-pulse" />
          )}
          <span className="text-xs font-medium text-light truncate">
            {label}
          </span>
        </div>

        {/* Value */}
        <div className="flex items-baseline gap-1 justify-start">
          <span className="text-xl font-semibold text-light leading-none">
            {formattedValue}
          </span>
          {unit && (
            <span className="text-sm text-light/70">
              {unit}
            </span>
          )}
        </div>

        {/* Warning badge — in flow, self-end */}
        {hasWarning && warning?.term ? (
          <div className="flex justify-start mt-1">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-[0.6875rem] border truncate w-auto font-bold   
              `}
              style={{
                backgroundColor: `${warningStyles.bg}`,
                color: `var(--light)`,
                borderColor: warningStyles.border,
              }}
            >
              {warning.term}
            </span>
          </div>
        ) : null}
      </div>
    </>
  );
};

const StationWeatherMetricsCard = ({
  telemetryMetrics,
  stationData,
  selectedMetric,
  onMetricSelect,
}: StationMetricsCardProps) => {
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const metrics = useMetricConfig(telemetryMetrics);
  const recordDate = formatDate(stationData?.telemetry?.recordedAt, locale);
  const isLoading = !telemetryMetrics;

  if (isLoading) {
    return (
      <div className="mt-8">
        <div className="flex justify-between items-start mb-3 sm:mb-4">
          <div className="flex border-l-4 border-l-main pl-2 mb-2 sm:mb-4 gap-2 sm:gap-4 items-end">
            <Skeleton className="h-6 w-36 sm:h-7 sm:w-44 rounded-md" />
            <Skeleton className="h-4 w-28 sm:w-36 rounded-md" />
          </div>
          <Skeleton className="h-8 w-10 sm:h-9 sm:w-48 rounded-md" />
        </div>

        <div className="px-2 grid grid-cols-2 md:grid-cols-4 gap-2 md:gap-3">
          {Array.from({ length: 8 }).map((_, index) => (
            <div key={`metric-skeleton-${index}`}>
              <div className="hidden glass md:flex md:h-full md:flex-col items-center justify-center md:min-h-35 p-5 md:p-4">
                <div className="md:w-full w-1/3 space-y-2">
                  <Skeleton className="h-4 w-24 rounded-md" />
                  <Skeleton className="h-7 w-20 rounded-md" />
                  <Skeleton className="h-4 w-16 rounded-full" />
                </div>
              </div>

              <div className="flex glass md:hidden justify-center max-h-18 min-h-18 items-center rounded-lg border px-4 py-2">
                <div className="w-1/2 min-h-12 flex flex-col justify-center items-center">
                  <div className="w-full space-y-1.5">
                    <Skeleton className="h-4 w-24 rounded-md" />
                    <Skeleton className="h-3 w-14 rounded-full" />
                  </div>
                </div>
                <div className="w-1/2 flex justify-end">
                  <Skeleton className="h-7 w-20 rounded-md" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mt-8">
      <div className="flex justify-between items-start mb-3 sm:mb-4">
      <div className="flex border-l-4 border-l-main pl-2 mb-2 sm:mb-4 gap-2 sm:gap-4 items-end">
        <h2 className="text-base sm:text-lg md:text-xl font-semibold text-light">
          {t("dashboard.metrics.title")}
        </h2>
        <p className="text-xs sm:text-sm leading-4 text-light pb-1">{recordDate.formatted}</p>
      </div>
      <Link
        href="/terminologies"
        className="inline-flex items-center px-2 py-1 sm:px-3 sm:w-48 glass text-xs sm:text-sm hover:bg-main/40 text-light shrink-0"
        aria-label={t("dashboard.metrics.guidesAriaLabel")}
      >
        <Sparkles className="h-4 w-4 sm:mr-1" />
        <span className="hidden sm:inline">{t("dashboard.metrics.guides")}</span>
      </Link>
    </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 md:gap-3">
        {metrics.map((metric) => (
          <DataCard
            key={metric.key}
            metricKey={metric.key}
            label={metric.label}
            value={metric.value}
            displayValue={metric.displayValue}
            unit={metric.unit}
            icon={metric.icon}
            color={metric.color}
            warning={metric.warning}
            selectedMetric={selectedMetric}
            onMetricSelect={onMetricSelect}
          />
        ))}
      </div>
    </div>
  );
};

export default StationWeatherMetricsCard;
