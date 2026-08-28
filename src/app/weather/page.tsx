import StationWeatherContainer from "@/components/features/dashboard/station-weather-container";
import { telemetryService } from "@/services/telemetry.service";
import { findNearestStation } from "@/lib/utils/location";
import { findStationByLocation } from "@/lib/utils/stationLookup";
import { StationPublicInfo, TelemetryPublicDTO } from "@/types/telemetry";

export const dynamic = "force-dynamic";

export default async function WeatherPage({
  searchParams,
}: {
  searchParams: { lat?: string; lon?: string; location?: string };
}) {
  let stations: StationPublicInfo[] = [];
  let dashboardStations: TelemetryPublicDTO[] = [];
  let initialStationId: string | null = null;

  try {
    dashboardStations = await telemetryService.getDashboardStations();
    stations = dashboardStations.map((item) => item.station);

    // Priority 1: Check if location param is provided
    if (searchParams.location && stations.length) {
      const matchedStation = findStationByLocation(stations, searchParams.location);

      if (matchedStation) {
        initialStationId = matchedStation.stationPublicId;
      }
    }

    // Priority 2: Try to get location from lat/lon query params
    if (!initialStationId && searchParams.lat && searchParams.lon && stations.length) {
      const userLat = parseFloat(searchParams.lat);
      const userLon = parseFloat(searchParams.lon);

      if (!isNaN(userLat) && !isNaN(userLon)) {
        const nearest = findNearestStation(stations, userLat, userLon);
        initialStationId = nearest?.stationPublicId || null;
      }
    }
  } catch (error) {
    console.error("Failed to fetch stations:", error);
  }

  return (
    <StationWeatherContainer
      stations={stations}
      initialDashboardStations={dashboardStations}
      initialStationId={initialStationId}
    />
  );
}
