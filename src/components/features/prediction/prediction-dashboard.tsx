"use client";

import { useState, useEffect, useTransition, useCallback } from "react";
import { useTranslations } from "next-intl";
import { RefreshCw } from "lucide-react";
import PublicDashboardTopBar from "@/components/shared/public-dashboard-top-bar";
import type { StationPublicInfo } from "@/types/telemetry";
import type { PredictionPublicDTO, PredictionHorizon } from "@/types/prediction";
import { findNearestStation } from "@/lib/utils/location";
import PredictionHorizonSelector from "./prediction-horizon-selector";
import PredictionWeatherForecast from "./prediction-weather-forecast";
import PredictionPeakSummary from "./prediction-peak-summary";
import ScrollIndicator from "@/components/shared/scroll-indicator";

interface PredictionDashboardProps {
  initialData: PredictionPublicDTO;
  stations: StationPublicInfo[];
  hasError?: boolean;
}

export default function PredictionDashboard({
  initialData,
  stations,
  hasError = false,
}: PredictionDashboardProps) {
  const t = useTranslations("prediction");
  const [data, setData] = useState<PredictionPublicDTO>(initialData);
  const [horizon, setHorizon] = useState<PredictionHorizon>("24h");
  const [isPending, startTransition] = useTransition();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isLocating, setIsLocating] = useState(false);
  const [nearestStationId, setNearestStationId] = useState<string | null>(null);

  useEffect(() => {
    setData(initialData);
  }, [initialData]);

  const handleStationSelect = useCallback((stationId: string) => {
    startTransition(async () => {
      try {
        setIsRefreshing(true);
        const res = await fetch(
          `/api/prediction/station/${stationId}?horizon=${horizon}`
        );
        const json = await res.json();
        if (json.success && json.data) {
          setData(json.data);
        }
      } catch (err) {
        console.error("Failed to switch station prediction:", err);
      } finally {
        setIsRefreshing(false);
      }
    });
  }, [horizon]);

  const handleDetectNearest = useCallback(() => {
    if (typeof window === "undefined" || !("geolocation" in navigator)) {
      return;
    }
    setIsLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setIsLocating(false);
        const userLat = pos.coords.latitude;
        const userLon = pos.coords.longitude;
        const nearest = findNearestStation(stations, userLat, userLon);
        if (nearest) {
          setNearestStationId(nearest.stationPublicId);
          if (nearest.stationPublicId !== data.station.stationPublicId) {
            handleStationSelect(nearest.stationPublicId);
          }
        }
      },
      (err) => {
        setIsLocating(false);
        console.warn("Geolocation permission not granted or unavailable:", err.message);
      },
      { enableHighAccuracy: false, timeout: 6000, maximumAge: 300000 }
    );
  }, [stations, data.station.stationPublicId, handleStationSelect]);

  // Automatically detect nearest station on first client mount
  useEffect(() => {
    handleDetectNearest();
  }, [handleDetectNearest]);

  const handleHorizonChange = (newHorizon: PredictionHorizon) => {
    setHorizon(newHorizon);
    startTransition(async () => {
      try {
        setIsRefreshing(true);
        const res = await fetch(
          `/api/prediction/station/${data.station.stationPublicId}?horizon=${newHorizon}`
        );
        const json = await res.json();
        if (json.success && json.data) {
          setData(json.data);
        }
      } catch (err) {
        console.error("Failed to update prediction horizon:", err);
      } finally {
        setIsRefreshing(false);
      }
    });
  };

  const handleRefresh = async () => {
    try {
      setIsRefreshing(true);
      const res = await fetch(
        `/api/prediction/station/${data.station.stationPublicId}?horizon=${horizon}`
      );
      const json = await res.json();
      if (json.success && json.data) {
        setData(json.data);
      }
    } catch (err) {
      console.error("Failed to refresh prediction:", err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleScrollDown = () => {
    const detailsElem = document.getElementById("prediction-details");
    if (detailsElem) {
      detailsElem.scrollIntoView({ behavior: "smooth" });
    } else {
      window.scrollTo({
        top: window.innerHeight,
        behavior: "smooth",
      });
    }
  };

  return (
    <>
      <div className="relative mx-auto flex min-h-[100svh] w-full max-w-7xl flex-col justify-between px-5 py-4 sm:px-8 sm:py-6 md:px-12 md:py-6">
        {/* Top Header Navigation */}
        <header className="w-full z-20">
          <div className="rounded-2xl bg-[#F9F9F6]/60 backdrop-blur-lg">
            <div className="px-6 py-3 md:py-4">
              <PublicDashboardTopBar />
            </div>
          </div>
        </header>

        {/* Center Hero Forecast */}
        <main className="my-auto flex w-full flex-1 flex-col items-center justify-center -mt-5 md:-mt-6 py-1 gap-2.5 md:gap-3.5">
          {data.weatherForecast && (
            <PredictionWeatherForecast
              station={data.station}
              stations={stations}
              weather={data.weatherForecast}
              forecastPoints={data.forecast}
              horizon={horizon}
              thresholds={data.summary.thresholds}
              onStationSelect={handleStationSelect}
              nearestStationId={nearestStationId}
              onDetectNearest={handleDetectNearest}
              isLocating={isLocating}
            />
          )}

          {/* Compact & Clean Horizon Selector */}
          <div className="flex items-center gap-2 rounded-full border border-slate-950/10 bg-white/70 px-3.5 py-1.5 shadow-2xs backdrop-blur-md mt-1 md:mt-2">
            <PredictionHorizonSelector
              selectedHorizon={horizon}
              onSelectHorizon={handleHorizonChange}
            />

            <span className="h-3.5 w-px bg-slate-200" />

            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 disabled:opacity-50"
              title="Refresh forecast data"
            >
              <RefreshCw className={`h-3 w-3 ${isRefreshing ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">{isRefreshing ? "..." : t("refresh")}</span>
            </button>
          </div>
        </main>

        {/* Bottom Scroll Indicator */}
        <div className="relative w-full pb-2 z-20 flex justify-center">
          <ScrollIndicator show={true} onClick={handleScrollDown} />
        </div>
      </div>

      {/* Below the Fold: Peak Flood Crest & Upstream Watershed Summary */}
      <div className="w-full bg-gradient-to-b from-transparent via-black/5 to-black/10 -mt-6 pb-4">
        <PredictionPeakSummary
          summary={data.summary}
          forecast={data.forecast}
          station={data.station}
          horizon={horizon}
          onSelectHorizon={handleHorizonChange}
          suddenRainBurst={data.suddenRainBurst}
        />
      </div>
    </>
  );
}
