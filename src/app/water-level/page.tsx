import type { Metadata } from "next";

import WaterLevelDashboard from "@/components/features/water-level/water-level-dashboard";
import { findNearestStation } from "@/lib/utils/location";
import { findStationByLocation } from "@/lib/utils/stationLookup";
import { waterLevelService } from "@/services/water-level.service";
import type { StationPublicInfo } from "@/types/telemetry";
import type { WaterLevelHistoryMetricDataPoint, WaterLevelPublicDTO } from "@/types/water-level";
import CustomFooter from "@/components/shared/custom-footer";
import StationCtaBanner from "@/components/features/dashboard/shared/station-cta-banner";

export const dynamic = "force-dynamic";

const PH_TIME_ZONE = "Asia/Manila";
const PH_UTC_OFFSET_MS = 8 * 60 * 60 * 1000;
const DAY_MS = 24 * 60 * 60 * 1000;

export const metadata: Metadata = {
  title: "Water Level",
  description:
    "Current calculated water level with a compact trend chart and freshness context.",
};

function getPhilippineDateParts(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: PH_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);

  const getPart = (type: Intl.DateTimeFormatPartTypes) =>
    Number(parts.find((part) => part.type === type)?.value);

  return {
    year: getPart("year"),
    month: getPart("month"),
    day: getPart("day"),
  };
}

function getPhilippineStartOfDayUtc(date = new Date()) {
  const { year, month, day } = getPhilippineDateParts(date);
  return new Date(Date.UTC(year, month - 1, day) - PH_UTC_OFFSET_MS);
}

export default async function WaterLevelPage({
  searchParams,
}: {
  searchParams: { lat?: string; lon?: string; location?: string };
}) {
  const now = new Date();
  const startOfToday = getPhilippineStartOfDayUtc(now);
  const startOfYesterday = new Date(startOfToday.getTime() - DAY_MS);
  const endOfToday = new Date(startOfToday.getTime() + DAY_MS);
  let dashboardStations: WaterLevelPublicDTO[] = [];
  let stations: StationPublicInfo[] = [];
  let selectedStation: StationPublicInfo | null = null;
  let currentReading: WaterLevelPublicDTO["waterLevel"] = null;
  let history: WaterLevelHistoryMetricDataPoint[] = [];
  let hasCurrentError = false;
  let hasHistoryError = false;

  try {
    dashboardStations = await waterLevelService.getDashboardStations();
    stations = dashboardStations.map((item) => item.station);

    if (searchParams.location && stations.length) {
      selectedStation = findStationByLocation(
        stations,
        searchParams.location,
        "waterLevel"
      );
    }

    if (!selectedStation && searchParams.lat && searchParams.lon && stations.length) {
      const userLat = parseFloat(searchParams.lat);
      const userLon = parseFloat(searchParams.lon);

      if (!isNaN(userLat) && !isNaN(userLon)) {
        selectedStation = findNearestStation(stations, userLat, userLon);
      }
    }

    selectedStation = selectedStation ?? stations[0] ?? null;
    currentReading =
      dashboardStations.find(
        (item) => item.station.stationPublicId === selectedStation?.stationPublicId
      )?.waterLevel ?? null;
  } catch (error) {
    console.error("Failed to fetch water-level stations:", error);
    hasCurrentError = true;
  }

  if (selectedStation) {
    try {
      const historyResult = await waterLevelService.getStationParameterHistory(
        selectedStation.stationPublicId,
        "distance",
        {
          interval: 60,
          startDate: startOfYesterday.toISOString(),
          endDate: endOfToday.toISOString(),
        }
      );

      history = historyResult.waterLevel;
      selectedStation = selectedStation ?? historyResult.station;
    } catch (error) {
      console.error("Failed to fetch water-level history:", error);
      hasHistoryError = true;
    }
  } else {
    hasHistoryError = true;
  }

  const todayHistory = history.filter(
    (point) => new Date(point.recordedAt).getTime() >= startOfToday.getTime()
  );
  const yesterdayHistory = history.filter(
    (point) => new Date(point.recordedAt).getTime() < startOfToday.getTime()
  );

  const fallbackCurrentValue = todayHistory.length
    ? todayHistory[todayHistory.length - 1]?.value ?? null
    : history[history.length - 1]?.value ?? null;
  const fallbackCurrentRecordedAt = todayHistory.length
    ? todayHistory[todayHistory.length - 1]?.recordedAt ?? null
    : history[history.length - 1]?.recordedAt ?? null;

  return (
    <>
      <WaterLevelDashboard
        station={selectedStation}
        stations={stations}
        currentValue={currentReading?.calculatedWaterLevel ?? fallbackCurrentValue}
        currentRecordedAt={currentReading?.recordedAt ?? fallbackCurrentRecordedAt}
        todayHistory={todayHistory}
        yesterdayHistory={yesterdayHistory}
        hasCurrentError={hasCurrentError}
        hasHistoryError={hasHistoryError}
      />

      <div className="snap-none">
        <StationCtaBanner />
        <CustomFooter />
      </div>
    </>
  );
}
