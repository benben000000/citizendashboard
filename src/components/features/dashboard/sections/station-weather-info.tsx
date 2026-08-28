"use client";

import { TelemetryPublicDTO, StationPublicInfo } from "@/types/telemetry";
import { useMemo, useState } from "react";
import StationWeatherMetricsCard from "../metrics/station-weather-metrics-card";
import StationWeatherMetricsChart from "../metrics/station-weather-metrics-chart";
import hardCodedStations from "@/lib/constants/stations.json";
import { StationMap } from "../../map/station-map";
import { StationSelectorPopover } from "@/components/shared/station-selector-popover";
import StationCtaBanner from "../shared/station-cta-banner";
import CustomFooter from "@/components/shared/custom-footer";
import { useTranslations } from "next-intl";

interface StationWeatherInfoProps {
  stationData: TelemetryPublicDTO | null;
  dashboardStations: TelemetryPublicDTO[];
  stations: StationPublicInfo[];
  selectedStationId: string | null;
  onStationSelect: (stationId: string) => void;
  nearestStationId?: string | null;
  onDetectNearest?: () => void;
  isLocating?: boolean;
}

const StationWeatherInfo = ({ 
  stationData, 
  dashboardStations,
  stations, 
  selectedStationId, 
  onStationSelect,
  nearestStationId,
  onDetectNearest,
  isLocating = false
}: StationWeatherInfoProps) => {
  const t = useTranslations();
  const [selectedMetric, setSelectedMetric] = useState<string>("temperature");

  // Reuse the shared dashboard payload so the map does not start its own telemetry poller.
  const mapStations = useMemo(
    () =>
      dashboardStations.length > 0
        ? dashboardStations
        : stations.map((station) => ({ station, telemetry: null })),
    [dashboardStations, stations]
  );

  const handleMetricSelect = (metricKey: string) => {
    setSelectedMetric(metricKey);
  }

  if (!stationData?.station) return null;

  const stationInfo = hardCodedStations.weather.stationIdToFetch.find(
    s => s.stationId === stationData.station.stationPublicId
  );

  return (
    <>
    <div className="px-4 pb-8 max-w-7xl mx-auto space-y-8">
      <div className="flex mt-6 md:mt-12 px-2">

        <div className="w-full">
          {/* Station Location */}
          <div className="flex flex-col justify-center">
            <div className="flex items-center relative">
              <StationSelectorPopover
                station={stationData.station}
                stations={stations}
                selectedStationId={selectedStationId}
                onStationSelect={onStationSelect}
                nearestStationId={nearestStationId}
                onDetectNearest={onDetectNearest}
                isLocating={isLocating}
              />
            </div>
          
          </div>



          {/* Weather Metrics */}
          <StationWeatherMetricsCard 
            telemetryMetrics={stationData.telemetry} 
            stationData={stationData}
            selectedMetric={selectedMetric} 
            onMetricSelect={handleMetricSelect} 
          />  
        </div>

      </div>

      {/* Historical Metric Chart */}
      <StationWeatherMetricsChart
        stationPublicId={stationData.station.stationPublicId}
        metricKey={selectedMetric}
      />

      <div>
        {stationInfo?.contactNumber && (
          <div>{t("dashboard.info.contact")}: {stationInfo.contactNumber}</div>
        )}
        
        {stationInfo?.email && (
          <div>{t("dashboard.info.email")}: {stationInfo.email}</div>
        )}
      </div>

      <div className="hidden md:block mt-4">
        <h2 className="border-l-4 border-l-main pl-2 text-light mb-4 text-lg md:text-xl font-semibold">
          {t("dashboard.info.exploreYourArea")}
        </h2>
        <div className="w-full h-[60vh] mt-4 rounded-lg overflow-hidden">
          <StationMap
            stations={mapStations}
            selectedStationId={selectedStationId}
            onStationSelect={onStationSelect}
            selectedMetric={selectedMetric}
            onMetricChange={handleMetricSelect}
          />
        </div>
      </div>

     

      {/* Terminology Redirect */}
      {/* <TerminologyRedirect /> */}
    
      {/* Disclaimer */}
      {/* <Disclaimer /> */}
    
    </div>

     <div className="snap-none">
        <StationCtaBanner />
        <CustomFooter />
      </div>

    </>
  )
}

export default StationWeatherInfo;
