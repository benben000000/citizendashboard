import type { WaterLevelHistoryMetricDataPoint } from "@/types/water-level";

type MaybeNumber = number | null | undefined;

export type WaterLevelTrend = "rising" | "falling" | "stable" | "unknown";

const WATER_LEVEL_NUMBER_FORMAT = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const WATER_LEVEL_TIME_FORMAT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

export function formatWaterLevelNumber(value: MaybeNumber): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }

  return WATER_LEVEL_NUMBER_FORMAT.format(value);
}

export function formatWaterLevelChange(value: MaybeNumber): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }

  const prefix = value > 0 ? "+" : "";
  return `${prefix}${formatWaterLevelNumber(value)} cm`;
}

export function getWaterLevelTrend(change: MaybeNumber): WaterLevelTrend {
  if (change === null || change === undefined || Number.isNaN(change)) {
    return "unknown";
  }

  if (Math.abs(change) < 0.02) {
    return "stable";
  }

  return change > 0 ? "rising" : "falling";
}

export function getWaterLevelTrendLabel(trend: WaterLevelTrend): string {
  switch (trend) {
    case "rising":
      return "Rising";
    case "falling":
      return "Falling";
    case "stable":
      return "Stable";
    default:
      return "No trend";
  }
}

export function formatWaterLevelTimestamp(dateString: MaybeNumber | string): string {
  if (typeof dateString !== "string" || !dateString) {
    return "No recent update";
  }

  const date = new Date(dateString);

  if (Number.isNaN(date.getTime())) {
    return "No recent update";
  }

  return WATER_LEVEL_TIME_FORMAT.format(date);
}

export function getMedianCadenceMinutes(
  readings: WaterLevelHistoryMetricDataPoint[]
): number | null {
  if (readings.length < 2) {
    return null;
  }

  const sorted = [...readings].sort(
    (left, right) => new Date(left.recordedAt).getTime() - new Date(right.recordedAt).getTime()
  );
  const intervals = sorted
    .map((reading, index) => {
      if (index === 0) {
        return null;
      }

      const previous = sorted[index - 1];
      const diffMinutes =
        (new Date(reading.recordedAt).getTime() - new Date(previous.recordedAt).getTime()) /
        (1000 * 60);

      return Number.isFinite(diffMinutes) && diffMinutes > 0 ? diffMinutes : null;
    })
    .filter((value): value is number => value !== null)
    .sort((left, right) => left - right);

  if (!intervals.length) {
    return null;
  }

  const midIndex = Math.floor(intervals.length / 2);
  const middleValue = intervals[midIndex];

  if (intervals.length % 2 === 0) {
    return Math.round((intervals[midIndex - 1] + middleValue) / 2);
  }

  return Math.round(middleValue);
}

export function formatCadenceLabel(minutes: number | null): string {
  if (minutes === null) {
    return "Cadence not available";
  }

  if (minutes < 60) {
    return `About every ${minutes} min`;
  }

  const hours = minutes / 60;
  if (hours < 24 && Number.isInteger(hours)) {
    return hours === 1 ? "About every hour" : `About every ${hours} hours`;
  }

  if (hours < 24) {
    return `About every ${hours.toFixed(1)} hours`;
  }

  const days = hours / 24;
  return days === 1 ? "About every day" : `About every ${days.toFixed(1)} days`;
}