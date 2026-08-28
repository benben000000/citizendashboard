"use client";

import React from "react";
import { Marker } from "react-map-gl/mapbox";
import { StationPublicInfo, TelemetryMetrics } from "@/types/telemetry";
import { getWarningInfo, getWarningStyles } from "@/lib/utils/weatherWarningUtil";
import {
  getParameterByKey,
  getTelemetryMetricValue,
  getWarningMetricKey,
} from "@/lib/constants/parameters";

interface StationWithTelemetry {
  station: StationPublicInfo;
  telemetry: TelemetryMetrics | null;
}

interface StationMarkersProps {
  stations: StationWithTelemetry[];
  selectedMetric: string;
  selectedStationId: string | null;
  onStationSelect: (stationId: string) => void;
}

const darkenHex = (hex: string, factor = 0.85): string => {
  const match = /^#?([a-fA-F0-9]{2})([a-fA-F0-9]{2})([a-fA-F0-9]{2})$/.exec(hex);
  if (!match) return hex;
  const r = Math.max(0, Math.min(255, Math.floor(parseInt(match[1], 16) * factor)));
  const g = Math.max(0, Math.min(255, Math.floor(parseInt(match[2], 16) * factor)));
  const b = Math.max(0, Math.min(255, Math.floor(parseInt(match[3], 16) * factor)));
  const toHex = (n: number) => n.toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};

const formatValue = (value: number | null, unit: string): string => {
  if (value === null || value === undefined) return "--";
  return `${value.toFixed(1)}${unit ? ` ${unit}` : ""}`;
};

export const StationMarkers: React.FC<StationMarkersProps> = ({
  stations,
  selectedMetric,
  selectedStationId,
  onStationSelect,
}) => {
  const sortedStations = React.useMemo(() => {
    if (!selectedStationId) return stations;
    return [...stations].sort((a, b) => {
      const aSelected = a.station.stationPublicId === selectedStationId ? 1 : 0;
      const bSelected = b.station.stationPublicId === selectedStationId ? 1 : 0;
      return aSelected - bSelected;
    });
  }, [stations, selectedStationId]);

  return (
    <>
      {sortedStations.map(({ station, telemetry }) => {
        const value = getTelemetryMetricValue(selectedMetric, telemetry);
        const unit = getParameterByKey(selectedMetric)?.unit ?? "";
        const displayValue = formatValue(value, unit);

        const info = getWarningInfo(getWarningMetricKey(selectedMetric), value);
        const warningStyles = getWarningStyles(info.color || undefined);
        const ringColor = info.color ? darkenHex(info.color, 0.85) : darkenHex("#ffffff", 0.85);

        let textColor;
        if (info.color === "#FBF300") {
          textColor = darkenHex(info.color, 0.5);
        } else if (info.color === null) {
          textColor = "#334155";
        } else {
          textColor = info.color ? darkenHex(info.color, 0.6) : "#ffffff";
        }

        const isSelected = station.stationPublicId === selectedStationId;

        return (
          <Marker
            key={station.stationPublicId}
            longitude={station.location[0]}
            latitude={station.location[1]}
          >
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onStationSelect(station.stationPublicId);
              }}
              className="flex flex-col items-center bg-transparent border-none p-0 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-sky-400 transition-transform duration-200 hover:scale-105"
              aria-label={`Select ${station.stationName} station`}
            >
              <div className="flex flex-col items-center leading-none relative">
                {isSelected && (
                  <div
                    className="absolute inset-0 rounded-md station-marker-pulse"
                    style={{
                      boxShadow: `0 0 0 0.125rem ${ringColor}`,
                    }}
                  />
                )}
                <div className="bg-white rounded-md">
                  <div
                    className={`
                      relative inline-flex items-center justify-center
                      font-semibold font-inter tracking-tight
                      rounded-md px-2 py-1
                      border border-transparent
                      transition-all duration-200
                      ${isSelected ? "ring-2 ring-offset-0" : "hover:ring-1 hover:ring-white/40"}
                      text-[0.875rem] md:text-[1.125rem]
                      shadow-sm hover:shadow-md
                    `}
                    style={{
                      borderColor: warningStyles.border,
                      backgroundColor: warningStyles.bg,
                      color: textColor,
                      ["--tw-ring-color" as any]: isSelected ? ringColor : undefined,
                    }}
                  >
                    {displayValue}
                  </div>
                </div>
              </div>
            </button>
          </Marker>
        );
      })}
    </>
  );
};
