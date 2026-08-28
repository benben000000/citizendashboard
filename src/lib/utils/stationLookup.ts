import type { StationPublicInfo } from "@/types/telemetry";
import stationLookupData from "@/lib/constants/stations.json";

interface StationLookup {
  weather: {
    stationIdToFetch: Array<{
      stationId: string;
      contactNumber: string;
      email: string;
      location: string;
    }>;
  };
  waterLevel: {
    stationIdToFetch: Array<{
      stationId: string;
      contactNumber: string;
      email: string;
      location: string;
    }>;
  };
}

type StationLookupGroup = keyof StationLookup;

// Cache the lookup maps for better performance
const locationToStationIdMaps: Partial<Record<StationLookupGroup, Map<string, string>>> = {};
const stationIdToLocationMaps: Partial<Record<StationLookupGroup, Map<string, string>>> = {};

export function initializeLookupMaps(group: StationLookupGroup = "weather") {
  if (!locationToStationIdMaps[group]) {
    const lookupData = stationLookupData as StationLookup;
    const locationToStationIdMap = new Map<string, string>();
    const stationIdToLocationMap = new Map<string, string>();
    
    lookupData[group].stationIdToFetch.forEach(item => {
      // Map location name to stationId from JSON
      locationToStationIdMap.set(item.location.toLowerCase(), item.stationId);
      // Also map stationId to location name for reverse lookup
      stationIdToLocationMap.set(item.stationId, item.location);
    });

    locationToStationIdMaps[group] = locationToStationIdMap;
    stationIdToLocationMaps[group] = stationIdToLocationMap;
  }
}

export function getStationIdByLocation(
  location: string,
  group: StationLookupGroup = "weather"
): string | null {
  initializeLookupMaps(group);
  const normalizedLocation = location.toLowerCase().trim();
  return locationToStationIdMaps[group]?.get(normalizedLocation) || null;
}

export function getLocationByStationId(
  stationId: string,
  group: StationLookupGroup = "weather"
): string | null {
  initializeLookupMaps(group);
  return stationIdToLocationMaps[group]?.get(stationId) || null;
}

export function findStationByLocation(
  stations: StationPublicInfo[], 
  location: string,
  group: StationLookupGroup = "weather"
): StationPublicInfo | null {
  // Step 1: Get the stationId from the JSON using the location
  const stationIdFromJson = getStationIdByLocation(location, group);
  
  if (stationIdFromJson) {
    // Step 2: Find the station in the stations array where stationPublicId matches
    const station = stations.find(
      station => station.stationPublicId === stationIdFromJson
    );
    
    if (station) {
      return station;
    }
  }
  
  // Fallback: try to match by city field directly (if needed)
  const normalizedLocation = location.toLowerCase().trim();
  return stations.find(
    station => station.city?.toLowerCase() === normalizedLocation
  ) || null;
}

// Optional: Get all available locations from the JSON
export function getAllAvailableLocations(): string[] {
  initializeLookupMaps();
  return Array.from(locationToStationIdMaps.weather?.keys() || []);
}
