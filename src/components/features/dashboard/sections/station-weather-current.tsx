import { Skeleton } from "@/components/ui/skeleton";
import { formatDate } from "@/lib/utils/date";
import { TelemetryPublicDTO, StationPublicInfo } from "@/types/telemetry";
import { getWeatherCondition } from "@/lib/utils/weather";
import WeatherIcon from "@/components/shared/weather-icons";
import RainyDayOverlay from "@/components/shared/rainy-day-overlay-window";
import { StationSelectorPopover } from "@/components/shared/station-selector-popover";
import { getWarningStyles } from "@/lib/utils/weatherWarningUtil";
import { AlertTriangle, TriangleAlert } from "lucide-react";
import type { ComponentType } from "react";
import { useLocale, useTranslations } from "next-intl";
import type { Locale } from "@/lib/i18n/translations";
import {
  STATION_WEATHER_METRIC_GROUPS,
  type StationWeatherMetricGroup,
} from "@/lib/config/stationWeatherMetricGroups";
import {
  useStationWeatherCurrentMetrics,
} from "@/hooks/useStationWeatherCurrentMetrics";
import type { WeatherMetricConfig } from "@/hooks/useWeatherMetric";
import { useStationPrecipitationHistory } from "@/hooks/useStationPrecipitationHistory";

interface OtherParamItem {
  key: string;
  icon?: ComponentType<{ className?: string }>;
  label: string;
  displayValue?: string;
  value: number | string | null | undefined;
  unit?: string;
  warning?: { color: string; term: string } | null;
}

interface OtherParams {
  params: OtherParamItem[];
}

const OtherParams = ({ params }: OtherParams) => {
  return (
    <>
      {/* Desktop View */}
      <div className="hidden md:block w-full md:-mt-20">
        <div className="grid grid-cols-2 gap-2 md:gap-5">
          {params.map((param) => {
            const hasWarning = !!param.warning;
            const warningStyles = getWarningStyles(param.warning?.color);
            const formattedValue =
              param.displayValue ??
              (typeof param.value === "number" ? param.value.toFixed(1) : param.value ?? "0");

            return (
              <div
                key={param.key}
                className="glass flex min-h-42 flex-col items-center justify-center p-5 md:p-4"
              >
                <div className="md:w-full w-1/3 flex items-center gap-2 md:mb-2">
                  <div className="relative">
                    {param.icon && <param.icon className="h-4 w-4 md:h-5 md:w-5 text-light" />}
                    {hasWarning && (
                      <AlertTriangle
                        className="absolute -top-1.5 -right-1.5 w-3 h-3 animate-pulse"
                        style={{ color: "var(--light)" }}
                      />
                    )}
                  </div>
                  <span className="text-base text-light font-medium">{param.label}</span>
                </div>

                <div className="md:w-full w-1/3 mt-2 flex items-center gap-1 text-light md:items-baseline justify-start">
                  <span className="text-xl md:text-2xl font-bold leading-none">{formattedValue}</span>
                  {param.unit && <span className="text-sm font-medium">{param.unit}</span>}
                </div>

                {hasWarning && param.warning?.term && (
                  <div
                    className="md:w-full w-1/3 inline-flex items-center justify-start rounded-full py-0.5 text-sm md:text-base font-bold"
                    style={{
                      backgroundColor: `${warningStyles.text}15`,
                      color: "var(--light)",
                    }}
                  >
                    {param.warning.term}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Mobile View */}
      <div className="md:hidden w-full mt-4">
        <div className="grid grid-cols-2 gap-2">
        {params.map((param) => {
          return (
            <div
              key={param.key}
              className="flex flex-col items-start justify-between"
            >
              <p className="text-sm font-light">{param.label}</p>
              <p className="text-base font-semibold">{typeof param.value === "number" ? param.value.toFixed(1) : param.value ?? "0"} {param.unit}</p>
            </div>
          );
        })}
        </div>
      </div>
    </>
  );
}

interface HeroMetricProps {
  metric: WeatherMetricConfig | undefined;
  condition: ReturnType<typeof getWeatherCondition>;
  recordedAt: string;
  todayAtLabel: string;
  suggestedAction?: string;
}

const formatMetricValue = (
  metric: Pick<WeatherMetricConfig, "displayValue" | "value"> | undefined,
) => {
  if (!metric) return "--";
  if (metric.displayValue) return metric.displayValue;
  if (typeof metric.value === "number") return Math.round(metric.value * 10) / 10;
  return metric.value ?? "--";
};

const hasPrecipitation = (telemetry: TelemetryPublicDTO["telemetry"] | undefined) => {
  if (!telemetry) return false;

  const precipitation = telemetry.precipitation ?? 0;
  const hourlyPrecipitation =
    telemetry.hourlyPrecip ??
    (telemetry as TelemetryPublicDTO["telemetry"] & { hourlyPrecipitation?: number | null })
      ?.hourlyPrecipitation ??
    0;

  return precipitation > 0 || hourlyPrecipitation > 0;
};

const HeroMetric = ({
  metric,
  condition,
  recordedAt,
  todayAtLabel,
  suggestedAction,
}: HeroMetricProps) => {
  const warning = metric?.warning;

  const hasValue = metric?.value != null || !!metric?.displayValue;
  const unit = metric?.unit;
  const actionText = suggestedAction ?? warning?.suggestionAction;

  return (
    <div className="flex items-center w-full">
      {hasValue ? (
        <div className="flex flex-col items-start justify-center gap-2">
          <p className="text-base font-medium text-light/90 text-left md:text-xl lg:text-2xl">
            <strong>{metric?.label}</strong>{" "}
            <span className="font-light">{todayAtLabel}</span>{" "}
            <span className="font-medium">{recordedAt}</span>
          </p>

          <div className="flex items-center gap-2">
            <p className="flex gap-2 text-[clamp(4rem,14vw,10rem)] font-bold text-light tabular-nums tracking-tighter leading-[1.03]">
              {formatMetricValue(metric)}
              {unit && (
                <span className="mt-2 self-start text-2xl font-bold leading-none tracking-tight text-light md:mt-4 md:text-5xl lg:mt-5 lg:text-6xl">
                  {unit}
                </span>
              )}
            </p>

            <WeatherIcon condition={condition} className="ml-2 h-16 w-16 shrink-0 items-center justify-center md:ml-4 md:h-24 md:w-24 lg:ml-6 lg:h-28 lg:w-28" />
          </div>

          {warning && (
            <>
              <div
                className="flex w-fit items-center justify-end rounded-full px-5 py-1.5 text-sm font-semibold md:px-7 md:text-lg"
                style={{ backgroundColor: warning.color, color: "white" }}
              >
                <TriangleAlert className="mr-2.5 h-5 w-5 animate-pulse" />
                <p>{warning.warningLevel}</p>
              </div>
              {actionText && (
                <p className="w-full max-w-xl py-2 text-base font-normal text-light/95 md:text-lg">
                  {actionText}
                </p>
              )}
            </>
          )}
        </div>
      ) : (
        <span className="text-6xl font-bold leading-none tracking-tighter text-light/70 md:text-[8rem]">
          --
        </span>
      )}
    </div>
  );
};


interface StationWeatherCurrentProps {
  stationData: TelemetryPublicDTO | null;
  stations: StationPublicInfo[];
  selectedStationId: string | null;
  onStationSelect: (stationId: string) => void;
  nearestStationId?: string | null;
  onDetectNearest?: () => void;
  isLocating?: boolean;
  metricGroup?: StationWeatherMetricGroup;
}

const StationWeatherCurrent = ({
  stationData,
  stations,
  selectedStationId,
  onStationSelect,
  nearestStationId,
  onDetectNearest,
  isLocating = false,
  metricGroup = STATION_WEATHER_METRIC_GROUPS.drySeason,
}: StationWeatherCurrentProps) => {
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const telemetryData = stationData?.telemetry;
  const stationInfo = stationData?.station;
  const condition = telemetryData ? getWeatherCondition(telemetryData) : "sunny";
  const showRainOverlay = hasPrecipitation(telemetryData);
  const { dayPrecipitation } = useStationPrecipitationHistory();

  const { heroMetric, otherParams, suggestedAction } = useStationWeatherCurrentMetrics(
    telemetryData,
    metricGroup,
    dayPrecipitation,
  );
  const recordAt = formatDate(telemetryData?.recordedAt, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }, locale);

  if (!telemetryData || !stationInfo) {
    return (
      <div className="flex h-1/3 w-full items-center justify-center md:h-3/5 px-2">
        <div className="flex h-full w-full flex-col justify-center items-start gap-4 md:gap-10">
          {/* Station selector skeleton */}
          <div className="flex flex-col gap-1.5">
            <Skeleton className="h-3 w-28 rounded-md" />
            <Skeleton className="h-9 w-56 md:h-10 md:w-80 rounded-md" />
          </div>

          <div className="flex w-full flex-col md:flex-row md:items-start md:gap-8 lg:gap-10">
            {/* Left: primary metric content */}
            <div className="flex h-full w-full flex-row items-center gap-2 md:flex-1 md:gap-5 lg:gap-6">
              <div className="flex flex-col items-start justify-center gap-2 w-full">
                <Skeleton className="h-5 w-60 md:h-7 md:w-96 rounded-md" />

                <div className="flex items-center gap-2">
                  <Skeleton className="h-24 w-48 md:h-40 md:w-80 rounded-lg" />
                  <Skeleton className="h-16 w-16 md:h-24 md:w-24 lg:h-28 lg:w-28 rounded-full" />
                </div>

                <Skeleton className="h-8 w-44 md:h-10 md:w-56 rounded-full" />
                <Skeleton className="h-4 w-full max-w-xl rounded-md" />
                <Skeleton className="h-4 w-5/6 max-w-lg rounded-md" />
              </div>
            </div>

            {/* Right panel is desktop-only in the real UI */}
            <div className="hidden h-full w-full items-center md:mt-0 md:flex md:max-w-102 lg:max-w-108">
              <div className="w-full p-4 md:p-5">
                <div className="grid grid-cols-2 gap-2 md:gap-5">
                  <Skeleton className="min-h-30 md:min-h-42 rounded-xl" />
                  <Skeleton className="min-h-30 md:min-h-42 rounded-xl md:translate-y-6" />
                  <Skeleton className="min-h-30 md:min-h-42 rounded-xl" />
                  <Skeleton className="min-h-30 md:min-h-42 rounded-xl md:translate-y-6" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-1/3 w-full items-center justify-center md:h-3/5 px-2">
      {showRainOverlay && (
        <RainyDayOverlay
          intensity={
            (telemetryData.hourlyPrecip ?? telemetryData.precipitation ?? 0) >= 2
              ? "steady"
              : "light"
          }
        />
      )}
      <div className="flex h-full w-full flex-col justify-center items-start gap-4 md:gap-10">

        <div className="">
          <StationSelectorPopover
            station={stationData.station}
            stations={stations}
            selectedStationId={selectedStationId}
            onStationSelect={onStationSelect}
            nearestStationId={nearestStationId}
            onDetectNearest={onDetectNearest}
            isLocating={isLocating}
            locationClassName="text-2xl md:text-4xl"
            showAddress={false}
          />
        </div>

        <div className="flex w-full flex-col md:flex-row md:items-start md:gap-8 lg:gap-10">

          <div className="flex h-full w-full flex-row items-center gap-2 md:flex-1 md:gap-5 lg:gap-6">

            <HeroMetric
              metric={heroMetric}
              condition={condition}
              recordedAt={recordAt.formatted}
              todayAtLabel={t("dashboard.current.todayAt")}
              suggestedAction={suggestedAction}
            />

          </div>

          {/* Secondary metrics */}
          <div className="mt-4 h-full flex items-center w-full md:mt-0 md:max-w-102 lg:max-w-108">
            <OtherParams params={otherParams} />
          </div>

        </div>
      </div>
    </div>
  );
};

export default StationWeatherCurrent;
