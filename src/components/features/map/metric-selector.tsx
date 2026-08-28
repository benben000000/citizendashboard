"use client";
import React from "react";
import {
  Thermometer,
  Droplets,
  Gauge,
  Wind,
  CloudRain,
  Eye,
  Sun,
  ThermometerSun,
  SunDim,
} from "lucide-react";
import { useTranslations } from "next-intl";
import type { TranslationKey } from "@/lib/i18n/translations";

interface Metric {
  key: string;
  labelKey: TranslationKey;
  icon: React.ElementType;
}

const METRICS: Metric[] = [
  { key: "temperature", labelKey: "common.metrics.temperature", icon: ThermometerSun },
  { key: "heatIndex", labelKey: "common.metrics.heatIndex", icon: Thermometer },
  { key: "humidity", labelKey: "common.metrics.humidity", icon: Droplets },
  { key: "pressure", labelKey: "common.metrics.pressure", icon: Gauge },
  { key: "windSpeed", labelKey: "common.metrics.wind", icon: Wind },
  { key: "precipitation", labelKey: "common.metrics.precipitation", icon: CloudRain },
  { key: "uvIndex", labelKey: "common.metrics.uvIndex", icon: SunDim },
  { key: "lightIntensity", labelKey: "common.metrics.light", icon: Eye },
];

interface MetricSelectorProps {
  selectedMetric: string;
  onMetricChange: (metric: string) => void;
}

export const MetricSelector: React.FC<MetricSelectorProps> = ({
  selectedMetric,
  onMetricChange,
}) => {
  const t = useTranslations();

  return (
    <div
      className="absolute top-2 left-2 z-10 rounded-xl p-1.5 flex flex-row gap-0.5 flex-wrap max-w-[calc(100vw-1rem)] sm:max-w-none bg-white/75 backdrop-blur-[1rem] backdrop-saturate-180 border border-black/10 shadow-[0_0.125rem_1rem_rgba(0,0,0,0.06),_0_0.0625rem_0.25rem_rgba(0,0,0,0.04)]"
    >
      {METRICS.map(({ key, labelKey, icon: Icon }) => {
        const label = t(labelKey);
        return (
        <button
          key={key}
          title={label}
          onClick={() => onMetricChange(key)}
          className={`p-1.5 sm:p-2 rounded-lg border-none cursor-pointer flex items-center justify-center transition-all duration-200 ${
            selectedMetric === key
              ? "bg-main text-light shadow-sm"
              : "bg-white/50 text-gray-600 hover:bg-white hover:text-gray-900 hover:shadow-sm"
          }`}
        >
          <Icon size={16} className="sm:w-4.5 sm:h-4.5" />
        </button>
        );
      })}
    </div>
  );
};
