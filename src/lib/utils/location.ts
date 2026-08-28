import type { StationPublicInfo } from "@/types/telemetry";

export function findNearestStation(
  stations: StationPublicInfo[],
  userLat: number,
  userLon: number
): StationPublicInfo | null {
  if (!Number.isFinite(userLat) || !Number.isFinite(userLon)) {
    return null;
  }

  let nearestStation: StationPublicInfo | null = null;
  let smallestDistance = Number.POSITIVE_INFINITY;

  for (const station of stations) {
    const [lon, lat] = station.location;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    const distance = haversineDistanceKm(userLat, userLon, lat, lon);
    if (distance < smallestDistance) {
      smallestDistance = distance;
      nearestStation = station;
    }
  }

  return nearestStation;
}

function haversineDistanceKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const toRad = (value: number) => (value * Math.PI) / 180;

  const R = 6371; // Earth radius in km
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}
