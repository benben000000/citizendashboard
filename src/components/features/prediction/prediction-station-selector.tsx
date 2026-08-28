"use client";

import { StationSelectorPopover } from "@/components/shared/station-selector-popover";
import type { StationPublicInfo } from "@/types/telemetry";

interface PredictionStationSelectorProps {
  station: StationPublicInfo | null;
  stations: StationPublicInfo[];
}

export default function PredictionStationSelector({
  station,
  stations,
}: PredictionStationSelectorProps) {
  if (!station) {
    return (
      <div className="text-sm text-slate-500">
        Station unavailable
      </div>
    );
  }

  return (
    <StationSelectorPopover
      station={station}
      stations={stations}
      selectedStationId={station.stationPublicId}
      onStationSelect={() => undefined}
      showAddress={true}
      stationGroup="waterLevel"
      locationClassName="text-2xl md:text-3xl"
      triggerClassName="w-full"
    />
  );
}
