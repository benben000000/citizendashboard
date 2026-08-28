import { useEffect, useState, useCallback, useRef } from "react";
import type { StationPublicInfo } from "@/types/telemetry";
import { findNearestStation } from "@/lib/utils/location";
import { getLocationByStationId } from "@/lib/utils/stationLookup";

/**
 * Owns station selection and browser geolocation behavior. It respects an
 * initial URL-selected station, can find the nearest station on demand, keeps
 * selection valid when the station list changes, and reports location errors.
 */

/** Returns station selection state and the action for detecting the nearest station. */
export function useNearestStation(
  stations: StationPublicInfo[], 
  initialStationId: string | null = null
) {
  const [selectedStationId, setSelectedStationId] = useState<string>(() => {
    // Priority 1: Use initialStationId from server (URL param)
    if (initialStationId) {
      return initialStationId;
    }

    // Priority 2: Fallback to first station
    return stations[0]?.stationPublicId ?? "";
  });
  
  const [nearestStationId, setNearestStationId] = useState<string | null>(
    initialStationId || null
  );
  const [isLocating, setIsLocating] = useState(false);
  const [locationError, setLocationError] = useState<string | null>(null);
  
  // Track if we've already done initial auto-selection
  const hasAutoSelectedRef = useRef(false);
  // Track if the user has manually changed stations
  const hasUserSelectedRef = useRef(!!initialStationId); // If URL provided location, consider that as user selection

  // Preference persistence removed: do not store preferred station in localStorage

  // Keep selected station in sync if the list of stations changes
  useEffect(() => {
    if (!stations.length) {
      setSelectedStationId("");
      return;
    }

    if (!selectedStationId) {
      setSelectedStationId(stations[0].stationPublicId);
      return;
    }

    const stillExists = stations.some(
      (station) => station.stationPublicId === selectedStationId
    );

    if (!stillExists) {
      setSelectedStationId(stations[0].stationPublicId);
    }
  }, [stations, selectedStationId]);


const detectNearestStation = useCallback(() => {
  if (!stations.length) return;

  if (typeof navigator === "undefined" || !navigator.geolocation) {
    setLocationError("Location is not available in this browser.");
    return;
  }

  setIsLocating(true);
  setLocationError(null);

  navigator.geolocation.getCurrentPosition(
    (position) => {
      const userLat = position.coords.latitude;
      const userLon = position.coords.longitude;

      const nearest = findNearestStation(stations, userLat, userLon);

      if (nearest) {
        setNearestStationId(nearest.stationPublicId);
        setSelectedStationId(nearest.stationPublicId);
        
        // Update URL with location
        if (typeof window !== "undefined") {
          const url = new URL(window.location.href);
          const locationParam = getLocationByStationId(nearest.stationPublicId);
          if (locationParam) {
            url.searchParams.set("location", locationParam);
            url.searchParams.delete("lat");
            url.searchParams.delete("lon");
            window.history.pushState({}, "", url.toString());
          }
        }
        
        // Mark that user has selected (they clicked the location button)
        hasUserSelectedRef.current = true;
      }

      setIsLocating(false);
    },
    (error) => {
      setIsLocating(false);

      if (error.code === error.PERMISSION_DENIED) {
        setLocationError(
          "We couldn't access your location. You can still choose a station from the list."
        );
      } else {
        setLocationError("Unable to detect your location right now.");
      }
    },
    {
      enableHighAccuracy: false,
      timeout: 10000,
      maximumAge: 5 * 60 * 1000,
    }
  );
}, [stations]);

  const handleStationChange = useCallback((stationId: string) => {
    setSelectedStationId(stationId);
    setLocationError(null);
    // Mark that user has manually selected a station
    hasUserSelectedRef.current = true;
  }, []);

  // Auto-select nearest station only once on initial load if no user preference exists
  useEffect(() => {
    // Only run this effect once
    if (hasAutoSelectedRef.current) return;

    // If server provided an initial station ID, prefer it.
    if (initialStationId) {
      hasAutoSelectedRef.current = true;
      setNearestStationId(initialStationId);
      if (!hasUserSelectedRef.current) {
        setSelectedStationId(initialStationId);
      }
      return;
    }

    // No persisted preference exists anymore—auto-detect location once on first load.
    if (stations.length > 0 && !hasAutoSelectedRef.current) {
      hasAutoSelectedRef.current = true;
      if (typeof navigator !== "undefined" && navigator.geolocation) {
        detectNearestStation();
      }
    }
  }, [stations, initialStationId, detectNearestStation]);

  return {
    selectedStationId,
    setSelectedStationId: handleStationChange,
    nearestStationId,
    detectNearestStation,
    isLocating,
    locationError,
  } as const;
}
