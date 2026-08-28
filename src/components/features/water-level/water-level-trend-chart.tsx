"use client";

import {
  CartesianGrid,
  Customized,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import CustomChartTooltip from "@/components/ui/custom-chart-tooltip";
import { ChartContainer } from "@/components/ui/chart";
import { useTheme } from "@/contexts/theme-context";
import { getIntlDateLocale } from "@/lib/utils/date";
import type { Locale } from "@/lib/i18n/translations";
import { useLocale, useTranslations } from "next-intl";

import WaterLevelBridgeLayer, { BRIDGE_REFERENCE_COLOR } from "./water-level-bridge-layer";

const TODAY_COLOR = "#38bdf8";
const YESTERDAY_COLOR = "#fbd008";
const Y_AXIS_TICK_INTERVAL = 200;
const BRIDGE_REFERENCE_TOP_PADDING = 100;
const WHOLE_NUMBER_FORMAT = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

export interface WaterLevelChartPoint {
  recordedAt: string;
  value: number | null;
}

interface WaterLevelTrendChartProps {
  todayData: WaterLevelChartPoint[];
  yesterdayData: WaterLevelChartPoint[];
  referenceThreshold?: number | null;
  waterLevelLabel: string;
  notEnoughHistoryLabel: string;
  bridgeReferenceLabel: string;
}

interface OverlayChartPoint {
  time: string;
  hourIndex: number;
  today: number | null;
  yesterday: number | null;
}

function buildOverlayData(
  todayData: WaterLevelChartPoint[],
  yesterdayData: WaterLevelChartPoint[],
  intlLocale: string
): OverlayChartPoint[] {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
  const startOfTomorrow = new Date(startOfToday.getTime() + 24 * 60 * 60 * 1000);
  const startOfYesterday = new Date(startOfToday.getTime() - 24 * 60 * 60 * 1000);

  const chartData: OverlayChartPoint[] = Array.from({ length: 24 }, (_, hour) => {
    const time = new Date(startOfToday.getTime() + hour * 60 * 60 * 1000);
    return {
      time: time.toLocaleTimeString(intlLocale, {
        hour: "numeric",
        minute: "2-digit",
      }),
      hourIndex: hour,
      today: null,
      yesterday: null,
    };
  });

  todayData.forEach((point) => {
    const date = new Date(point.recordedAt);
    if (date < startOfToday || date >= startOfTomorrow) return;
    const hour = date.getHours();
    chartData[hour].today = typeof point.value === "number" ? point.value : null;
  });

  yesterdayData.forEach((point) => {
    const date = new Date(point.recordedAt);
    if (date < startOfYesterday || date >= startOfToday) return;
    const hour = date.getHours();
    chartData[hour].yesterday = typeof point.value === "number" ? point.value : null;
  });

  return chartData;
}

function buildYAxisScale(
  chartData: OverlayChartPoint[],
  referenceThreshold?: number | null
): { domain: [number, number]; ticks: number[] } {
  const values = chartData.flatMap((point) =>
    [point.today, point.yesterday].filter((value): value is number => typeof value === "number")
  );

  const hasReferenceThreshold =
    typeof referenceThreshold === "number" && Number.isFinite(referenceThreshold);

  if (hasReferenceThreshold) {
    values.push(referenceThreshold);
  }

  const minValue = Math.min(...values);
  const maxValue = hasReferenceThreshold
    ? Math.max(...values, referenceThreshold + BRIDGE_REFERENCE_TOP_PADDING)
    : Math.max(...values);
  let domainMin = Math.floor(minValue / Y_AXIS_TICK_INTERVAL) * Y_AXIS_TICK_INTERVAL;
  let domainMax =
    Math.ceil(maxValue / Y_AXIS_TICK_INTERVAL) * Y_AXIS_TICK_INTERVAL +
    Y_AXIS_TICK_INTERVAL;

  if (domainMin === domainMax) {
    domainMin -= Y_AXIS_TICK_INTERVAL;
    domainMax += Y_AXIS_TICK_INTERVAL;
  }

  const tickCount = Math.round((domainMax - domainMin) / Y_AXIS_TICK_INTERVAL) + 1;
  const ticks = Array.from(
    { length: tickCount },
    (_, index) => domainMin + index * Y_AXIS_TICK_INTERVAL
  );

  return {
    domain: [domainMin, domainMax],
    ticks,
  };
}

export default function WaterLevelTrendChart({
  todayData,
  yesterdayData,
  referenceThreshold,
  waterLevelLabel,
  notEnoughHistoryLabel,
  bridgeReferenceLabel,
}: WaterLevelTrendChartProps) {
  const { theme } = useTheme();
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const chartData = buildOverlayData(todayData, yesterdayData, getIntlDateLocale(locale));
  const gridStroke = theme === "dark" ? "rgba(226, 232, 240, 0.45)" : "var(--border)";
  const bridgeReferenceColor = theme === "dark" ? "#ffffff" : BRIDGE_REFERENCE_COLOR;
  const chartConfig = {
    today: {
      label: t("common.chart.today"),
      color: TODAY_COLOR,
    },
    yesterday: {
      label: t("common.chart.yesterday"),
      color: YESTERDAY_COLOR,
    },
  } as const;
  const hasReferenceThreshold =
    typeof referenceThreshold === "number" && Number.isFinite(referenceThreshold);
  const hasEnoughData =
    chartData.filter((point) => point.today !== null || point.yesterday !== null).length >= 2;

  if (!hasEnoughData) {
    return (
      <div className="flex min-h-[200px] items-center justify-center rounded-3xl border border-dashed border-border/60 bg-background/30 px-4 text-center text-sm text-muted-foreground md:min-h-[260px]">
        {notEnoughHistoryLabel}
      </div>
    );
  }

  const yAxisScale = buildYAxisScale(chartData, referenceThreshold);

  return (
    <div className="glass p-3">
      <ChartContainer config={chartConfig as never} className="aspect-auto h-[200px] w-full md:h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 24, right: 24, left: 0, bottom: 6 }}>
            <CartesianGrid strokeDasharray="4 8" stroke={gridStroke} opacity={0.75} />
            <XAxis
              dataKey="time"
              tickLine={false}
              axisLine={false}
              minTickGap={24}
              tickMargin={10}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
            />
            <YAxis
              width={56}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickFormatter={(value: number) => WHOLE_NUMBER_FORMAT.format(value)}
              domain={yAxisScale.domain}
              ticks={yAxisScale.ticks}
            />
            <Tooltip
              content={
                <CustomChartTooltip
                  parameterLabel={waterLevelLabel}
                  unit="cm"
                  todayDataKey="today"
                  yesterdayDataKey="yesterday"
                />
              }
            />
            {hasReferenceThreshold ? (
              <Customized
                component={
                  <WaterLevelBridgeLayer
                    referenceValue={referenceThreshold}
                    referenceColor={bridgeReferenceColor}
                  />
                }
              />
            ) : null}
            <Line
              type="monotone"
              dataKey="today"
              name={t("common.chart.today")}
              stroke="var(--color-today)"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 5, strokeWidth: 0 }}
            />
            <Line
              type="monotone"
              dataKey="yesterday"
              name={t("common.chart.yesterday")}
              stroke="var(--color-yesterday)"
              strokeWidth={2}
              strokeDasharray="6 6"
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>
      <div className="flex items-center justify-center gap-5 pt-3 text-xs font-medium text-muted-foreground">
        <span className="inline-flex items-center gap-2">
          <span
            className="h-0.5 w-7 rounded-full"
            style={{ backgroundColor: TODAY_COLOR }}
          />
          {t("common.chart.today")}
        </span>
        <span className="inline-flex items-center gap-2">
          <svg width="28" height="4" aria-hidden="true">
            <line
              x1="0"
              y1="2"
              x2="28"
              y2="2"
              stroke={YESTERDAY_COLOR}
              strokeWidth="2"
              strokeDasharray="6 4"
            />
          </svg>
          {t("common.chart.yesterday")}
        </span>
        {hasReferenceThreshold ? (
          <span className="inline-flex items-center gap-2">
            <svg width="28" height="6" aria-hidden="true">
              <line
                x1="0"
                y1="3"
                x2="28"
                y2="3"
                stroke={bridgeReferenceColor}
                strokeWidth="2"
                strokeDasharray="8 5"
                strokeLinecap="round"
              />
            </svg>
            {bridgeReferenceLabel}
          </span>
        ) : null}
      </div>
    </div>
  );
}
