"use client";
import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import type { StationPublicInfo, TelemetryPublicDTO } from "@/types/telemetry";
import StationWeatherCurrent from "./sections/station-weather-current";
import ScrollIndicator from "@/components/shared/scroll-indicator";
import StationHeader from "./shared/station-header";
import { useStationData } from "@/hooks/useStationData";
import StationWeatherInfo from "./sections/station-weather-info";
import { useNearestStation } from "@/hooks/useNearestStation";
import StationWeatherActions from "./sections/station-weather-actions";
import { getLocationByStationId } from "@/lib/utils/stationLookup";
import {
  getStationWeatherMetricGroupForDate,
  STATION_WEATHER_METRIC_GROUPS,
} from "@/lib/config/stationWeatherMetricGroups";
import { StationPrecipitationHistoryProvider } from "@/hooks/useStationPrecipitationHistory";

interface Props {
  stations: StationPublicInfo[];
  initialDashboardStations?: TelemetryPublicDTO[];
  initialStationId?: string | null;
}



const VISITED_KEY = "kloudtrack_has_seen_station_current";

export default function StationWeatherContainer({ 
  stations, 
  initialDashboardStations,
  initialStationId = null
}: Props) {
  const { 
    selectedStationId, 
    setSelectedStationId, 
    nearestStationId, 
    detectNearestStation, 
    isLocating, 
    locationError 
  } = useNearestStation(stations, initialStationId);
  
  const [showScrollIndicator, setShowScrollIndicator] = useState(true);
  const [headerTop, setHeaderTop] = useState(0);
  const [headerPosition, setHeaderPosition] = useState<"fixed" | "absolute">("fixed");
  const [snapMode, setSnapMode] = useState<"mandatory" | "proximity">("mandatory");
  const headerTopRef = useRef(headerTop);
  const headerPositionRef = useRef(headerPosition);
  const snapModeRef = useRef(snapMode);
  const scrollFrameRef = useRef<number | null>(null);
  const section1Ref = useRef<HTMLElement>(null);
  const section2Ref = useRef<HTMLElement>(null);
  const section3Ref = useRef<HTMLElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedStationId && typeof window !== "undefined") {
      const searchParams = new URLSearchParams(window.location.search);
      const currentLocation = searchParams.get("location");
      const locationFromStation = getLocationByStationId(selectedStationId);
      
      // If URL doesn't have the correct location param, update it
      if (locationFromStation && currentLocation !== locationFromStation) {
        const params = new URLSearchParams(window.location.search);
        params.set("location", locationFromStation);
        params.delete("lat");
        params.delete("lon");
        window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
      }
    }
  }, [selectedStationId, detectNearestStation]);

  useEffect(() => {
    const container = containerRef.current;

    const updateScrollState = () => {
      const section1 = section1Ref.current;
      const section2 = section2Ref.current;
      const section3 = section3Ref.current;

      if (!section1 || !section2 || !section3) return;

      const section3Top = section3.getBoundingClientRect().top;
      const nextSnapMode = section3Top <= 0 ? "proximity" : "mandatory";
      const nextHeaderPosition = section3Top <= 0 ? "absolute" : "fixed";
      const nextHeaderTop = section3Top <= 0 ? section3.offsetTop : 0;

      // When section 3 reaches the top, relax snapping for natural footer scrolling.
      if (snapModeRef.current !== nextSnapMode) {
        snapModeRef.current = nextSnapMode;
        setSnapMode(nextSnapMode);
      }

      // Only update React state when the computed header position actually changes.
      if (headerPositionRef.current !== nextHeaderPosition) {
        headerPositionRef.current = nextHeaderPosition;
        setHeaderPosition(nextHeaderPosition);
      }

      if (headerTopRef.current !== nextHeaderTop) {
        headerTopRef.current = nextHeaderTop;
        setHeaderTop(nextHeaderTop);
      }
    };

    const handleScroll = () => {
      if (scrollFrameRef.current !== null) return;

      // Batch scroll work to one state check per animation frame.
      scrollFrameRef.current = window.requestAnimationFrame(() => {
        scrollFrameRef.current = null;
        updateScrollState();
      });
    };

    if (container) {
      container.addEventListener("scroll", handleScroll);
      return () => {
        container.removeEventListener("scroll", handleScroll);
        if (scrollFrameRef.current !== null) {
          window.cancelAnimationFrame(scrollFrameRef.current);
        }
      };
    }
  }, []);

  const handleScrollToInfo = useCallback(() => {
    try {
      window.localStorage.setItem(VISITED_KEY, "true");
      setShowScrollIndicator(false);
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  const { data: stationData, dashboardStations } = useStationData(
    selectedStationId,
    initialDashboardStations
  );
  const currentMetricGroup = useMemo(() => getStationWeatherMetricGroupForDate(), []);
  const isWetSeason = currentMetricGroup === STATION_WEATHER_METRIC_GROUPS.wetSeason;

  const headerContent = (
    <div className="rounded-2xl bg-[#F9F9F6]/60 backdrop-blur-lg ">
      <div className="px-6 py-4">
        <StationHeader
          stations={stations}
          selectedId={selectedStationId}
          onSelect={setSelectedStationId}
          onUseLocation={detectNearestStation}
          isLocating={isLocating}
          locationError={locationError}
          nearestStationId={nearestStationId}
        />
      </div>
    </div>
  );

  return (
    <StationPrecipitationHistoryProvider
      stationId={selectedStationId}
      enabled={isWetSeason}
    >
    <div
      ref={containerRef}
      className={`h-screen h-[100svh] overflow-y-auto snap-y ${
        snapMode === "mandatory" ? "snap-mandatory" : "snap-proximity"
      } scroll-smooth relative`}
    >
      <div className="relative min-h-full ">
        <div
          className="z-50 px-4 pt-4 pb-2 max-w-360 w-full mx-auto"
          style={{
            position: headerPosition,
            top: headerTop,
            left: 0,
            right: 0,
          }}
        >
          {headerContent}
        </div>

        <section ref={section1Ref} className="relative h-screen h-[100svh] snap-start snap-always">
          <div className="px-4 pb-8 max-w-7xl mx-auto h-full flex items-center " style={{ paddingTop: "1rem" }}>
            <StationWeatherCurrent
              stationData={stationData}
              stations={stations}
              selectedStationId={selectedStationId}
              onStationSelect={setSelectedStationId}
              nearestStationId={nearestStationId}
              onDetectNearest={detectNearestStation}
              isLocating={isLocating}
              metricGroup={currentMetricGroup}
            />
            <ScrollIndicator show={showScrollIndicator} onClick={handleScrollToInfo} />
          </div>
        </section>

        <section ref={section2Ref} className="relative h-screen h-[100svh] snap-start snap-always">
          <div className="px-4 pb-8 max-w-7xl mx-auto h-full flex items-stretch" style={{ paddingTop: "6rem" }}>
            <StationWeatherActions
              stationData={stationData}
              stations={stations}
              selectedStationId={selectedStationId}
              onStationSelect={setSelectedStationId}
              nearestStationId={nearestStationId}
              onDetectNearest={detectNearestStation}
              isLocating={isLocating}
              className="h-full"
              metricGroup={currentMetricGroup}
            />
            <ScrollIndicator show={showScrollIndicator} onClick={handleScrollToInfo} />
          </div>
        </section>

        <section ref={section3Ref} className="min-h-screen min-h-[100svh] snap-start snap-always" style={{ paddingTop: "5rem" }}>
          <StationWeatherInfo
            stationData={stationData}
            dashboardStations={dashboardStations}
            stations={stations}
            selectedStationId={selectedStationId}
            onStationSelect={setSelectedStationId}
            nearestStationId={nearestStationId}
            onDetectNearest={detectNearestStation}
            isLocating={isLocating}
          />
          
        </section>


      </div>
    </div>
    </StationPrecipitationHistoryProvider>
  );
}
