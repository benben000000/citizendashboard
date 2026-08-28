"use client";

import { StationSelectorPopover } from "@/components/shared/station-selector-popover";
import type { StationPublicInfo } from "@/types/telemetry";
import { useTranslations } from "next-intl";

interface WaterLevelStationSelectorProps {
  station: StationPublicInfo | null;
  stations: StationPublicInfo[];
}

export default function WaterLevelStationSelector({
  station,
  stations,
}: WaterLevelStationSelectorProps) {
  const t = useTranslations();

  if (!station) {
    return (
      <div className="text-sm text-muted-foreground">
        {t("waterLevel.stationSelector.unavailable")}
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
