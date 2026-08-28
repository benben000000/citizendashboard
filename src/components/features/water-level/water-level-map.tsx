"use client";

import { useEffect, useRef, useState } from "react";
import Map, { Marker, type MapRef } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";

import { useTheme } from "@/contexts/theme-context";
import { initializeMapboxFixes } from "@/lib/utils/mapbox-config";
import { formatWaterLevelNumber } from "@/lib/utils/water-level";
import type { StationPublicInfo } from "@/types/telemetry";

interface WaterLevelMapProps {
  station: StationPublicInfo | null;
  currentValue: number | null;
  tokenMissingLabel: string;
  stationFallbackLabel: string;
}

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
}

export default function WaterLevelMap({
  station,
  currentValue,
  tokenMissingLabel,
  stationFallbackLabel,
}: WaterLevelMapProps) {
  const { theme } = useTheme();
  const mapRef = useRef<MapRef>(null);
  const stationLocation = station?.location;
  const [viewState, setViewState] = useState<ViewState>(() => ({
    longitude: stationLocation?.[0] ?? 120.9842,
    latitude: stationLocation?.[1] ?? 14.5995,
    zoom: stationLocation ? 12 : 7,
  }));

  const mapboxToken = process.env.NEXT_PUBLIC_MAPBOX_API_KEY;
  const stationLabel = station?.stationName ?? stationFallbackLabel;
  const currentLabel = `${formatWaterLevelNumber(currentValue)} cm`;
  const mapStyle =
    theme === "dark"
      ? "mapbox://styles/mapbox/navigation-night-v1"
      : "mapbox://styles/mapbox/standard";

  const philippinesBounds: [[number, number], [number, number]] = [
    [116.9, 4.2],
    [127.0, 21.1],
  ];

  useEffect(() => {
    const cleanup = initializeMapboxFixes();
    return cleanup;
  }, []);

  useEffect(() => {
    if (!stationLocation) return;

    const targetZoom = 10;
    const nextView = {
      longitude: stationLocation[0],
      latitude: stationLocation[1],
      zoom: targetZoom,
    };

    setViewState(nextView);
    mapRef.current?.getMap().flyTo({
      center: [nextView.longitude, nextView.latitude],
      zoom: targetZoom,
      duration: 1500,
      essential: true,
    });
  }, [stationLocation]);

  if (!mapboxToken) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-background/40 px-6 text-center text-sm text-muted-foreground">
        {tokenMissingLabel}
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-hidden rounded-2xl">
      <Map
        ref={mapRef}
        mapboxAccessToken={mapboxToken}
        longitude={viewState.longitude}
        latitude={viewState.latitude}
        zoom={viewState.zoom}
        minZoom={4}
        maxZoom={18}
        maxBounds={philippinesBounds}
        onMove={(event) => {
          setViewState({
            longitude: event.viewState.longitude,
            latitude: event.viewState.latitude,
            zoom: event.viewState.zoom,
          });
        }}
        mapStyle={mapStyle}
        style={{ width: "100%", height: "100%" }}
        attributionControl={false}
      >
        {stationLocation ? (
          <Marker longitude={stationLocation[0]} latitude={stationLocation[1]}>
            <div
              className="flex flex-col items-center border-none bg-transparent p-0"
              aria-label={stationLabel}
            >
              <div className="flex flex-col items-center leading-none relative">
                <div className="bg-white rounded-md">
                  <div className="relative inline-flex items-center justify-center rounded-md border border-sky-300 bg-sky-50 px-2 py-1 text-[0.875rem] font-semibold tracking-tight text-sky-800 shadow-sm md:text-[1.125rem]">
                    {currentLabel}
                  </div>
                </div>
              </div>
            </div>
          </Marker>
        ) : null}
      </Map>
    </div>
  );
}
