"use client";

import { getIntlDateLocale } from "@/lib/utils/date";
import { useLocale, useTranslations } from "next-intl";
import type { Locale } from "@/lib/i18n/translations";

interface CustomChartTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
  parameterLabel: string;
  unit: string;
  todayDataKey?: string;
  yesterdayDataKey?: string;
}

const CustomChartTooltip = ({
  active,
  payload,
  label,
  parameterLabel,
  unit,
  todayDataKey = "value1",
  yesterdayDataKey = "value2",
}: CustomChartTooltipProps) => {
  const t = useTranslations();
  const locale = useLocale() as Locale;
  const intlLocale = getIntlDateLocale(locale);

  if (!active || !payload || !payload.length) return null;

  const entryPayload = payload[0]?.payload;
  const unitStr = unit ? ` ${unit}` : "";
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
  const startOfYesterday = new Date(startOfToday.getTime() - 24 * 60 * 60 * 1000);
  const hourIndex = entryPayload?.hourIndex;

  const timeHeader = (() => {
    if (typeof hourIndex === "number" && Number.isFinite(hourIndex)) {
      const time = new Date(startOfToday.getTime() + hourIndex * 60 * 60 * 1000);
      return time.toLocaleTimeString(intlLocale, { hour: "2-digit", minute: "2-digit" });
    }
    return label ? String(label) : "";
  })();

  const formatTooltipDate = (date: Date) =>
    date.toLocaleDateString(intlLocale, {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

  return (
    <div className="bg-white/95 border border-gray-300 rounded-lg shadow-xl p-4 min-w-50 backdrop-blur-sm">
      <p className="font-semibold text-black mb-3 text-sm border-b border-gray-300 pb-2">
        {timeHeader}
      </p>
      <p className="text-xs font-medium text-gray-600 mb-2">
        {parameterLabel}
      </p>
      <div className="flex flex-col gap-2">
        {payload
          .filter((item) => item?.value !== null && item?.value !== undefined)
          .map((item) => {
            const labelText =
              item?.name ||
              (item?.dataKey === todayDataKey || item?.dataKey === "today"
                ? t("common.chart.today")
                : item?.dataKey === yesterdayDataKey || item?.dataKey === "yesterday"
                  ? t("common.chart.yesterday")
                  : t("common.chart.value"));
            const rawValue = Number(item.value);
            const value = Number.isFinite(rawValue)
              ? Math.round(rawValue * 100) / 100
              : item.value;
            const dotColor = item.color || item.stroke || item.fill || "#111827";
            const baseDate =
              item?.dataKey === yesterdayDataKey || item?.dataKey === "yesterday"
                ? startOfYesterday
                : startOfToday;
            const dateLabel = formatTooltipDate(baseDate);

            return (
              <div key={item.dataKey} className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full shadow-lg"
                    style={{
                      backgroundColor: dotColor,
                      boxShadow: `0 0 0.5rem ${dotColor}`,
                    }}
                  />
                  <span className="text-xs font-medium text-gray-700">
                    {labelText}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-right">
                  <span className="text-xs font-semibold text-gray-900">
                    {value}{unitStr}
                  </span>
                  <span className="text-[0.7rem] text-gray-500">
                    {dateLabel}
                  </span>
                </div>
              </div>
            );
          })}
      </div>
    </div>
  );
};

export default CustomChartTooltip;
