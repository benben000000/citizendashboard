"use client";
import React, { useMemo, useCallback } from "react";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Area,
  AreaChart,
  Bar,
  BarChart,
  LabelList,
  Line,
  LineChart,
} from "recharts";
import { ParameterConfig } from "@/types/parameter";
import { Cloud } from "lucide-react";
import { TelemetryMetricRaw } from "@/types/telemetry-raw";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
} from "@/components/ui/chart";
import CustomChartTooltip from "@/components/ui/custom-chart-tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { useLocale, useTranslations } from "next-intl";
import { getIntlDateLocale } from "@/lib/utils/date";
import type { Locale } from "@/lib/i18n/translations";

// ─── types ────────────────────────────────────────────────────────────────────

interface ChartDataPoint {
  time: string;
  /** hour index 0-23, used for isolated-dot lookups */
  hourIndex: number;
  value1: number | null; // today
  value2: number | null; // yesterday
}

interface StationParameterChartProps {
  /** Today's data points (midnight → now) */
  todayData: TelemetryMetricRaw[];
  /** Yesterday's data points (full day) */
  yesterdayData: TelemetryMetricRaw[];
  parameter: ParameterConfig;
  loading: boolean;
  error: string | null;
  onRetry?: () => void;
}

// ─── helpers (unchanged from original) ────────────────────────────────────────

const getDomain = (key: string) => {
  if (key === "humidity") return [0, 100];
  if (key === "uvIndex") return [0, 12];
  if (key === "lightIntensity") {
    return ([, dataMax]: [number, number]) =>
      [0, Math.ceil((dataMax || 0) * 1.1)] as [number, number];
  }
  return undefined;
};

const getFillOpacity = (key: string): number => {
  if (key === "humidity") return 0.3;
  if (key === "windSpeed") return 0.4;
  return 0.5;
};

const getYAxisTickFormatter = (
  key: string
): ((v: number) => string) | undefined => {
  if (key === "lightIntensity") {
    return (v: number) =>
      v >= 1000 ? `${Math.round((v / 1000) * 10) / 10}k` : `${v}`;
  }
  return undefined;
};

// ─── today / yesterday colors ─────────────────────────────────────────────────

const TODAY_COLOR = "#ef4444";
const YESTERDAY_COLOR = "#3b82f6";

// ─── component ────────────────────────────────────────────────────────────────

const StationParameterChart: React.FC<StationParameterChartProps> = ({
  todayData,
  yesterdayData,
  parameter,
  loading,
  error,
  onRetry,
}) => {
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const intlLocale = getIntlDateLocale(locale);
  const chartType = parameter.chartType ?? "line";
  const gridColor = "var(--muted-foreground)";

  // ── build a fixed 24-hour timeline, map both datasets onto it ──────────────
  const chartData = useMemo<ChartDataPoint[]>(() => {
    const now = new Date();
    const startOfToday = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate(),
      0, 0, 0, 0
    );
    const startOfTomorrow = new Date(
      startOfToday.getTime() + 24 * 60 * 60 * 1000
    );
    const startOfYesterday = new Date(
      startOfToday.getTime() - 24 * 60 * 60 * 1000
    );

    // Build base 24-slot timeline
    const timeline: ChartDataPoint[] = Array.from({ length: 24 }, (_, i) => {
      const t = new Date(startOfToday.getTime() + i * 60 * 60 * 1000);
      return {
        time: t.toLocaleTimeString(intlLocale, { hour: "2-digit", minute: "2-digit" }),
        hourIndex: i,
        value1: null,
        value2: null,
      };
    });

    // Map today → value1 (guard: only accept points that actually fall on today)
    todayData.forEach((pt) => {
      const date = new Date(pt.recordedAt);
      if (date < startOfToday || date >= startOfTomorrow) return;
      const h = date.getHours();
      if (timeline[h]) {
        timeline[h].value1 =
          pt.value !== null && pt.value !== undefined ? Number(pt.value) : null;
      }
    });

    // Map yesterday → value2 (guard: only accept points that actually fall on yesterday)
    yesterdayData.forEach((pt) => {
      const date = new Date(pt.recordedAt);
      if (date < startOfYesterday || date >= startOfToday) return;
      const h = date.getHours();
      if (timeline[h]) {
        timeline[h].value2 =
          pt.value !== null && pt.value !== undefined ? Number(pt.value) : null;
      }
    });

    return timeline;
  }, [intlLocale, todayData, yesterdayData]);

  // ── isolated-dot renderer (same logic as HeatIndexChart) ──────────────────
  const renderIsolatedDot = useCallback(
    (dataKey: "value1" | "value2", color: string) =>
      function IsolatedDot(props: {
        cx: number;
        cy: number;
        index: number;
        value: number | null;
      }) {
        const { cx, cy, index, value } = props;
        if (value === null || value === undefined)
          return <g key={`empty-${dataKey}-${index}`} />;

        const prev = chartData[index - 1]?.[dataKey];
        const next = chartData[index + 1]?.[dataKey];
        const isIsolated = prev == null && next == null;

        if (!isIsolated) return <g key={`skip-${dataKey}-${index}`} />;
        return (
          <circle
            key={`dot-${dataKey}-${index}`}
            cx={cx}
            cy={cy}
            r={4}
            fill={color}
            stroke="none"
          />
        );
      },
    [chartData]
  );

  // ── HIGH / LOW markers (today only, value1) ────────────────────────────────
  const { maxIndex, minIndex, minCount } = useMemo(() => {
    const values = chartData
      .map((d) => d.value1)
      .filter((v): v is number => v !== null);
    if (values.length === 0) return { maxIndex: -1, minIndex: -1, minCount: 0 };
    const max = Math.max(...values);
    const min = Math.min(...values);
    const minCountValue = values.filter((v) => v === min).length;
    return {
      maxIndex: chartData.findIndex((d) => d.value1 === max),
      minIndex: chartData.findIndex((d) => d.value1 === min),
      minCount: minCountValue,
    };
  }, [chartData]);

  const renderExtremeLabel = (
    cx: number,
    anchorY: number,
    isMax: boolean,
    value: number
  ) => {
    const label = isMax ? t("common.chart.highestToday") : t("common.chart.lowestToday");
    const formatted = `${value.toFixed(1)}${parameter.unit ? ` ${parameter.unit}` : ""}`;

    // Estimate text widths based on character count + font size
    const labelWidth = label.length * 8;       // ~7.5px per char at 12px bold
    const formattedWidth = formatted.length * 14; // ~14px per char at 14px bold

    const accentW = 3;
    const paddingLeft = accentW + 8;
    const paddingRight = 12;

    const contentWidth = Math.max(labelWidth, formattedWidth);
    const boxW = paddingLeft + contentWidth + paddingRight;
    const boxH = 44;
    const boxX = cx - boxW / 2;
    const boxY = anchorY - boxH - 14;

    return (
      <g>
        <line
          x1={cx} y1={boxY + boxH} x2={cx} y2={anchorY - 6}
          stroke={TODAY_COLOR} strokeWidth={1} strokeDasharray="3 2" opacity={0.5}
        />
        <rect x={boxX} y={boxY} width={boxW} height={boxH} rx={4} fill="#ffffff" stroke="#e5e5e5" strokeWidth={0.5} />
        <rect x={boxX} y={boxY} width={accentW} height={boxH} fill={TODAY_COLOR} rx={1} />
        <text x={boxX + paddingLeft} y={boxY + 18} fill="#737373" fontSize={12} fontWeight="600" letterSpacing="0.08em" fontFamily="Inter, sans-serif">{label}</text>
        <text x={boxX + paddingLeft} y={boxY + 36} fill="#0a0a0a" fontSize={14} fontWeight="700" fontFamily="Inter, sans-serif">{formatted}</text>
      </g>
    );
  };

  const renderCustomDot = (props: any) => {
    const { cx, cy, index, payload } = props;
    if (index === maxIndex || index === minIndex) {
      return (
        <g key={`extreme-dot-${index}`}>
          <circle cx={cx} cy={cy} r={5} fill={TODAY_COLOR} stroke="#ffffff" strokeWidth={2} />
          {renderExtremeLabel(cx, cy, index === maxIndex, payload.value1 ?? payload.value)}
        </g>
      );
    }
    return <circle key={`dot-${index}`} cx={cx} cy={cy} r={0} fill="none" />;
  };

  const renderBarExtremeLabel = (props: any) => {
    const { x, y, width, index, value } = props;
    if (value === null || value === undefined) return null;
    if (index === maxIndex) {
      const cx = x + width / 2;
      return renderExtremeLabel(cx, y, true, Number(value));
    }
    if (index === minIndex && minCount === 1) {
      const cx = x + width / 2;
      return renderExtremeLabel(cx, y, false, Number(value));
    }
    return null;
  };

  // ── chart config ───────────────────────────────────────────────────────────
  const chartConfig = {
    value1: { label: t("common.chart.today"), color: TODAY_COLOR },
    value2: { label: t("common.chart.yesterday"), color: YESTERDAY_COLOR },
  } satisfies ChartConfig;

  // ── loading / error / empty states ────────────────────────────────────────
  if (loading) {
    return (
      <div className="hidden md:block">
        <h2 className="border-l-4 border-l-main pl-2 text-light mb-4 text-lg md:text-xl font-semibold">
          {t("dashboard.chart.title")}
        </h2>
        <div className="md:pl-4 md:pr-8 pl-0 pr-4 py-4 glass rounded-lg border">
          <div className="h-48 md:h-72 p-2 md:p-3">
            <div className="h-full w-full rounded-md border border-border/50 p-3 md:p-4">
              <Skeleton className="h-full w-full rounded-md" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-[#F9F9F6]/60 border-2 border-card-border p-8 hidden md:block">
        <div className="flex flex-col items-center justify-center h-100">
          <div className="border-2 border-red-500 bg-red-500/10 p-4 mb-4">
            <Cloud size={28} className="text-red-500" strokeWidth={2} />
          </div>
          <span className="text-red-400 text-xs text-center max-w-75 mb-6">[ERROR] {error}</span>
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-6 py-2 border-2 border-input-border bg-secondary hover:bg-secondary-hover text-muted-foreground hover:text-light text-xs uppercase tracking-wider"
            >
            {t("common.status.retry")}
            </button>
          )}
        </div>
      </div>
    );
  }

  const hasAnyData = chartData.some((d) => d.value1 !== null || d.value2 !== null);
  if (!hasAnyData) {
    return (
      <div className="bg-[#F9F9F6]/60 border-2 border-card-border p-8 hidden md:block">
        <div className="flex flex-col items-center justify-center h-100">
          <div className="border-2 border-card-border p-4 mb-4">
            <Cloud size={28} className="text-muted-foreground" strokeWidth={2} />
          </div>
          <span className="text-muted-foreground text-xs uppercase tracking-wider">
            {t("common.status.noDataAvailable")}
          </span>
        </div>
      </div>
    );
  }

  // ── shared axis props ──────────────────────────────────────────────────────
  const commonXAxisProps = {
    dataKey: "time" as const,
    tick: { fontSize: 11 },
    tickMargin: 6,
    interval: 2,
    tickLine: false,
    axisLine: false,
    className: "text-muted-foreground",
  };

  const commonYAxisProps = {
    domain: getDomain(parameter.key) as any,
    tick: { fontSize: 12 },
    tickFormatter: getYAxisTickFormatter(parameter.key) as any,
    tickLine: false,
    axisLine: false,
    className: "text-muted-foreground",
  };

  const tooltipContent = (
    <CustomChartTooltip
      parameterLabel={parameter.label}
      unit={parameter.unit || ""}
    />
  );

  // ── chart renderer ─────────────────────────────────────────────────────────
  const renderChart = () => {
    const commonProps = {
      accessibilityLayer: true,
      data: chartData,
      margin: { top: 50, right: 35, left: -10, bottom: 10 },
    };

    switch (chartType) {
      case "area":
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
            <XAxis {...commonXAxisProps} />
            <YAxis {...commonYAxisProps} />
            <ChartTooltip content={tooltipContent} />
            {/* Today — solid, on top */}
            <Area
              type="monotone"
              dataKey="value1"
              stroke={TODAY_COLOR}
              fill={TODAY_COLOR}
              fillOpacity={getFillOpacity(parameter.key)}
              strokeWidth={2}
              dot={renderCustomDot}
              activeDot={{ r: 5, fill: TODAY_COLOR, stroke: "#ffffff", strokeWidth: 2 }}
              name={t("common.chart.today")}
            />
            {/* Yesterday — dashed, above for visibility */}
            <Area
              type="monotone"
              dataKey="value2"
              stroke={YESTERDAY_COLOR}
              fill={YESTERDAY_COLOR}
              fillOpacity={0.1}
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={renderIsolatedDot("value2", YESTERDAY_COLOR)}
              activeDot={{ r: 4 }}
              name={t("common.chart.yesterday")}
            />
          </AreaChart>
        );

      case "bar":
        return (
          <BarChart {...commonProps} barGap={6} barCategoryGap={16}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
            <XAxis {...commonXAxisProps} />
            <YAxis {...commonYAxisProps} />
            <ChartTooltip content={tooltipContent} />
            <Bar
              dataKey="value1"
              fill={TODAY_COLOR}
              maxBarSize={20}
              isAnimationActive
              name={t("common.chart.today")}
            >
              <LabelList content={renderBarExtremeLabel} />
            </Bar>
            <Bar
              dataKey="value2"
              fill={YESTERDAY_COLOR}
              maxBarSize={20}
              isAnimationActive
              name={t("common.chart.yesterday")}
            />
          </BarChart>
        );

      default: // line
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
            <XAxis {...commonXAxisProps} />
            <YAxis {...commonYAxisProps} />
            <ChartTooltip content={tooltipContent} />
            {/* Today — solid with HIGH/LOW markers */}
            <Line
              type="monotone"
              dataKey="value1"
              stroke={TODAY_COLOR}
              strokeWidth={2}
              dot={renderCustomDot}
              activeDot={{ r: 5, fill: TODAY_COLOR, stroke: "#ffffff", strokeWidth: 2 }}
              name={t("common.chart.today")}
            />
            {/* Yesterday — dashed, above for visibility */}
            <Line
              type="monotone"
              dataKey="value2"
              stroke={YESTERDAY_COLOR}
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={renderIsolatedDot("value2", YESTERDAY_COLOR)}
              activeDot={{ r: 4 }}
              name={t("common.chart.yesterday")}
            />
          </LineChart>
        );
    }
  };

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div className="hidden md:block">
      <h2 className="border-l-4 border-l-main pl-2 text-light mb-4 text-lg md:text-xl font-semibold">
        {t("dashboard.chart.title")}
      </h2>
      <div className="md:pl-4 md:pr-8 pl-0 pr-4 py-4 glass rounded-lg border">
        <div className="h-48 md:h-72">
          <ChartContainer
            config={chartConfig}
            className="h-full w-full overflow-visible"
          >
            {renderChart()}
          </ChartContainer>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 pt-3 pb-1">
          <div className="flex items-center gap-2">
            <span
              className="inline-block w-8 h-0.5 rounded"
              style={{ backgroundColor: TODAY_COLOR }}
            />
            <span className="text-xs text-muted-foreground font-medium">{t("common.chart.today")}</span>
          </div>
          <div className="flex items-center gap-2">
            <svg width="32" height="4" className="shrink-0">
              <line
                x1="0" y1="2" x2="32" y2="2"
                stroke={YESTERDAY_COLOR}
                strokeWidth="2"
                strokeDasharray="5 3"
              />
            </svg>
            <span className="text-xs text-muted-foreground font-medium">{t("common.chart.yesterday")}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StationParameterChart;
