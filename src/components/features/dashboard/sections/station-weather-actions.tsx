"use client";

import { TriangleAlert } from "lucide-react";
import { StationSelectorPopover } from "@/components/shared/station-selector-popover";
import { formatDate } from "@/lib/utils/date";
import { cn } from "@/lib/utils/cn";
import { TelemetryPublicDTO, StationPublicInfo } from "@/types/telemetry";
import { useLocale, useTranslations } from "next-intl";
import type { Locale } from "@/lib/i18n/translations";
import {
  STATION_WEATHER_METRIC_GROUPS,
  type StationWeatherMetricGroup,
} from "@/lib/config/stationWeatherMetricGroups";
import { useStationWeatherActionMetrics } from "@/hooks/useStationWeatherActionMetrics";
import type { WeatherMetricConfig } from "@/hooks/useWeatherMetric";
import { useStationPrecipitationHistory } from "@/hooks/useStationPrecipitationHistory";

interface StationWeatherActionsProps {
  stationData: TelemetryPublicDTO | null;
  stations: StationPublicInfo[];
  selectedStationId: string | null;
  onStationSelect: (stationId: string) => void;
  nearestStationId?: string | null;
  onDetectNearest?: () => void;
  isLocating?: boolean;
  className?: string;
  metricGroup?: StationWeatherMetricGroup;
}

const formatMetricValue = (
  metric: Pick<WeatherMetricConfig, "displayValue" | "key" | "value"> | undefined,
) => {
  if (!metric) return "--";
  if (metric.displayValue) return metric.displayValue;
  if (typeof metric.value === "number") {
    return metric.value.toFixed(1);
  }
  return metric.value ?? "--";
};

const StationWeatherActions = ({
  stationData,
  stations,
  selectedStationId,
  onStationSelect,
  nearestStationId,
  onDetectNearest,
  isLocating = false,
  className,
  metricGroup = STATION_WEATHER_METRIC_GROUPS.drySeason,
}: StationWeatherActionsProps) => {
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const telemetry = stationData?.telemetry;
  const { dayPrecipitation } = useStationPrecipitationHistory();
  const {
    actionItems,
    actionMetric,
    stats,
    warning,
    warningStyles,
  } = useStationWeatherActionMetrics(telemetry, metricGroup, locale, dayPrecipitation);

  if (!stationData?.station) return null;

  return (
    <div
      className={cn(
        "flex h-full w-full flex-col gap-2 py-2 md:gap-5 md:py-5",
        className,
      )}
    >
      <div className="flex w-full gap-2 flex-row md:items-start md:justify-between">
        <div className="w-full flex items-center">
          <StationSelectorPopover
            station={stationData.station}
            stations={stations}
            selectedStationId={selectedStationId}
            onStationSelect={onStationSelect}
            nearestStationId={nearestStationId}
            onDetectNearest={onDetectNearest}
            isLocating={isLocating}
            showAddress={true}
            locationClassName="text-base md:text-2xl"
          />
        </div>
      </div>

      <div className="flex gap-2">
        <div
          className="glass w-full md:w-1/2 rounded-xl border px-3 py-2.5 flex gap-2 md:rounded-2xl md:px-5 md:py-4 flex-row md:items-stretch md:justify-between"
          style={{
            borderColor: warningStyles.border,
          }}
        >
          <div className="flex flex-col flex-1 h-24 justify-between gap-1.5">
            <p className="text-xs font-medium uppercase tracking-widest opacity-75 md:text-[0.6875rem]">
              {actionMetric?.label}
            </p>
            <p className="text-5xl font-medium leading-none md:text-7xl">
              {formatMetricValue(actionMetric)}
              {actionMetric?.unit && (
                <span className="text-base md:text-xl">{actionMetric.unit}</span>
              )}
            </p>
          </div>

          <div className="flex flex-col justify-between gap-1.5 items-end">
            <div className="w-full md:w-auto md:shrink-0 md:text-right">
              <p className="mt-0.5 text-xs text-light text-right md:text-sm">
                {formatDate(telemetry?.recordedAt, locale).formatted}
              </p>
            </div>

            {warning && (
              <div
                className="flex w-fit items-center justify-end rounded-full px-5 py-1.5 text-xs font-semibold md:px-7 md:text-sm"
                style={{ backgroundColor: warning.color, color: "white" }}
              >
                <TriangleAlert className="mr-2.5 h-5 w-5 animate-pulse" />
                <p>{warning.warningLevel}</p>
              </div>
            )}
          </div>
        </div>

        <div className="hidden md:flex md:w-1/2 gap-2">
          {stats.map((metric) => (
            <div
              key={metric.key}
              className="flex flex-col w-full gap-0.5 glass p-4 h-full"
            >
              <p className="text-[0.625rem] font-medium uppercase tracking-wide text-light/60 md:text-[0.6875rem]">
                {metric.label}
              </p>
              <p className="text-4xl font-medium leading-none text-light flex items-end h-full">
                {formatMetricValue(metric)}
                {metric.unit && (
                  <span className="text-xs font-normal text-light/50 ml-0.5">{metric.unit}</span>
                )}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="h-px w-full bg-border/40" />

      <div className="flex flex-1 flex-col gap-2">
        <p className="md:text-sm font-medium uppercase tracking-wide text-light/60 text-[0.6875rem]">
          {t("dashboard.actions.stepsToStaySafe")}
        </p>

        <div className={cn("grid grid-cols-1", "glass")}>
          {actionItems.map(({ icon: Icon, text }, i) => {
            const SafeIcon = Icon as React.ElementType;

            return (
              <div key={i} className={cn("flex items-center gap-2 p-3")}>
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-background/60 md:h-8 md:w-8 md:rounded-lg">
                  <SafeIcon
                    className="h-4 w-4 md:h-5 md:w-5"
                    style={{ color: "var(--foreground)" }}
                  />
                </div>

                <div className="flex flex-col gap-0.5">
                  <p className="text-sm leading-snug text-light/90 md:text-sm md:leading-relaxed">{text}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default StationWeatherActions;
