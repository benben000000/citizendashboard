import type { Metadata } from "next";
import PredictionDashboard from "@/components/features/prediction/prediction-dashboard";
import { findNearestStation } from "@/lib/utils/location";
import { findStationByLocation } from "@/lib/utils/stationLookup";
import { waterLevelService } from "@/services/water-level.service";
import { telemetryService } from "@/services/telemetry.service";
import { predictionService } from "@/services/prediction.service";
import { DEFAULT_CENTRAL_LUZON_STATIONS } from "@/lib/constants/default-stations";
import type { StationPublicInfo } from "@/types/telemetry";
import type { PredictionPublicDTO } from "@/types/prediction";
import CustomFooter from "@/components/shared/custom-footer";
import StationCtaBanner from "@/components/features/dashboard/shared/station-cta-banner";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "LNN Water Level Prediction & Flood Forecasting | Kloudtrack",
  description:
    "Continuous-time hydrological forecasting using Liquid Neural Networks (LNN) from real-time weather and river station telemetry.",
};

export default async function PredictionPage({
  searchParams,
}: {
  searchParams: { lat?: string; lon?: string; location?: string };
}) {
  let stations: StationPublicInfo[] = [];
  let selectedStation: StationPublicInfo | null = null;
  let predictionData: PredictionPublicDTO | null = null;
  let hasError = false;

  try {
    // Fetch all active stations across both water level and weather telemetry networks
    const [waterStations, weatherStations] = await Promise.all([
      waterLevelService.getDashboardStations().catch(() => []),
      telemetryService.getDashboardStations().catch(() => []),
    ]);

    const allStationsMap = new Map<string, StationPublicInfo>();
    DEFAULT_CENTRAL_LUZON_STATIONS.forEach((item) => allStationsMap.set(item.stationPublicId, item));
    waterStations.forEach((item) => allStationsMap.set(item.station.stationPublicId, item.station));
    weatherStations.forEach((item) => allStationsMap.set(item.station.stationPublicId, item.station));

    stations = Array.from(allStationsMap.values());

    if (searchParams.location && stations.length) {
      selectedStation =
        findStationByLocation(stations, searchParams.location, "weather") ||
        findStationByLocation(stations, searchParams.location, "waterLevel");
    }

    if (!selectedStation && searchParams.lat && searchParams.lon && stations.length) {
      const userLat = parseFloat(searchParams.lat);
      const userLon = parseFloat(searchParams.lon);

      if (!isNaN(userLat) && !isNaN(userLon)) {
        selectedStation = findNearestStation(stations, userLat, userLon);
      }
    }

    selectedStation = selectedStation ?? stations[0] ?? DEFAULT_CENTRAL_LUZON_STATIONS[0];

    const stationId = selectedStation?.stationPublicId || "O3z0j5bG";
    predictionData = await predictionService.getPredictionForStation(stationId, "24h");
  } catch (error) {
    console.error("Failed to load prediction dashboard:", error);
    hasError = true;
    stations = DEFAULT_CENTRAL_LUZON_STATIONS;
    predictionData = await predictionService.getPredictionForStation("O3z0j5bG", "24h");
  }

  return (
    <>
      {predictionData && (
        <PredictionDashboard
          initialData={predictionData}
          stations={stations}
          hasError={hasError}
        />
      )}

      <div className="snap-none">
        <StationCtaBanner />
        <CustomFooter />
      </div>
    </>
  );
}
