"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import Map from "react-map-gl/mapbox";
import type { MapRef } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import { MetricSelector } from "./metric-selector";
import { StationMarkers } from "./station-markers";
import { StationPublicInfo, TelemetryMetrics } from "@/types/telemetry";
import { initializeMapboxFixes } from "@/lib/utils/mapbox-config";
import { useTranslations } from "next-intl";

interface StationWithTelemetry {
  station: StationPublicInfo;
  telemetry: TelemetryMetrics | null;
}

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
}

interface StationMapProps {
  stations: StationWithTelemetry[];
  selectedStationId: string | null;
  onStationSelect: (stationId: string) => void;
  selectedMetric?: string;
  onMetricChange?: (metric: string) => void;
  initialViewState?: Partial<ViewState>;
}

export const StationMap: React.FC<StationMapProps> = ({
  stations,
  selectedStationId,
  onStationSelect,
  selectedMetric: externalSelectedMetric,
  onMetricChange: externalOnMetricChange,
  initialViewState,
}) => {
  const t = useTranslations();
  const mapRef = useRef<MapRef>(null);
  const [internalSelectedMetric, setInternalSelectedMetric] = useState<string>("heatIndex");
  const selectedMetric = externalSelectedMetric ?? internalSelectedMetric;
  const setSelectedMetric = externalOnMetricChange ?? setInternalSelectedMetric;
  const [viewState, setViewState] = useState<ViewState>({
    longitude: 120.9842, // Default to Philippines center
    latitude: 14.5995,
    zoom: 7,
    ...initialViewState,
  });

  // Calculate center from selected station or use default
  const mapCenter = useMemo(() => {
    if (selectedStationId) {
      const selected = stations.find(
        (s) => s.station.stationPublicId === selectedStationId
      );
      if (selected?.station.location) {
        return {
          longitude: selected.station.location[0],
          latitude: selected.station.location[1],
        };
      }
    }
    return null;
  }, [selectedStationId, stations]);

  // Initialize Mapbox fixes (disable telemetry, suppress warnings)
  useEffect(() => {
    const cleanup = initializeMapboxFixes();
    return cleanup;
  }, []);

  // Track previous station to only animate on selection change
  const prevStationIdRef = useRef<string | null>(null);

  // Animate map to selected station with smooth transition
  React.useEffect(() => {
    // Only animate if station actually changed and map is ready
    if (mapCenter && mapRef.current && selectedStationId !== prevStationIdRef.current) {
      const map = mapRef.current.getMap();
      const currentZoom = map.getZoom();
      const targetZoom = 10; 
      
      // Use flyTo for smooth animation
      map.flyTo({
        center: [mapCenter.longitude, mapCenter.latitude],
        zoom: targetZoom,
        duration: 1500, 
        essential: true, 
      });

      // Update viewState after animation starts
      setViewState((prev) => ({
        ...prev,
        longitude: mapCenter.longitude,
        latitude: mapCenter.latitude,
        zoom: targetZoom,
      }));

      // Update previous station ID
      prevStationIdRef.current = selectedStationId;
    } else if (!selectedStationId) {
      // Reset when no station is selected
      prevStationIdRef.current = null;
    }
  }, [mapCenter, selectedStationId]);

  const mapboxToken = process.env.NEXT_PUBLIC_MAPBOX_API_KEY;

  if (!mapboxToken) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-background/40">
        <span className="text-muted-foreground text-sm">
          {t("dashboard.map.tokenMissing")}
        </span>
      </div>
    );
  }

  // Philippines bounds: [southwest, northeast]
  // Southwest: [longitude, latitude] - westernmost and southernmost point
  // Northeast: [longitude, latitude] - easternmost and northernmost point
  const philippinesBounds: [[number, number], [number, number]] = [
    [116.9, 4.2],  // Southwest corner (west, south)
    [127.0, 21.1], // Northeast corner (east, north)
  ];

  return (
    <div className="relative w-full h-full">
      <Map
        ref={mapRef}
        mapboxAccessToken={mapboxToken}
        longitude={viewState.longitude}
        latitude={viewState.latitude}
        zoom={viewState.zoom}
        minZoom={4}
        maxZoom={18}
        maxBounds={philippinesBounds}
        onMove={(evt) => {
          setViewState({
            longitude: evt.viewState.longitude,
            latitude: evt.viewState.latitude,
            zoom: evt.viewState.zoom,
          });
        }}
        mapStyle="mapbox://styles/mapbox/standard"
        style={{ width: "100%", height: "100%" }}
        attributionControl={false}
      >
        <MetricSelector
          selectedMetric={selectedMetric}
          onMetricChange={setSelectedMetric}
        />
        <StationMarkers
          stations={stations}
          selectedMetric={selectedMetric}
          selectedStationId={selectedStationId}
          onStationSelect={onStationSelect}
        />
      </Map>
    </div>
  );
};
