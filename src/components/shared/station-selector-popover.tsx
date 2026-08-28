import { useState, useCallback } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import type { StationPublicInfo } from "@/types/telemetry";
import { MapPin, Home, LocateFixed, ChevronDown, Loader2, Check } from "lucide-react";
import { Popover, PopoverTrigger } from "@/components/ui/popover";
import * as PopoverPrimitive from "@radix-ui/react-popover";
import { cn } from "@/lib/utils/cn";
import { useTranslations } from "next-intl";
import { getLocationByStationId } from "@/lib/utils/stationLookup";

interface StationSelectorPopoverProps {
  station: StationPublicInfo;
  stations: StationPublicInfo[];
  selectedStationId: string | null;
  onStationSelect: (stationId: string) => void;
  nearestStationId?: string | null;
  onDetectNearest?: () => void;
  isLocating?: boolean;
  locationClassName?: string;
  showAddress?: boolean;
  stationGroup?: "weather" | "waterLevel";
  triggerClassName?: string;
}

export const StationSelectorPopover = ({
  station,
  stations,
  selectedStationId,
  onStationSelect,
  nearestStationId,
  onDetectNearest,
  isLocating = false,
  locationClassName,
  showAddress = true,
  stationGroup = "weather",
  triggerClassName,
}: StationSelectorPopoverProps) => {
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const t = useTranslations();
  
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const updateUrlWithStation = useCallback((stationId: string, stationLocation?: string) => {
    // Get location name from station lookup or fallback to city
    let locationParam = stationLocation || getLocationByStationId(stationId, stationGroup);
    
    if (!locationParam) {
      // Fallback: find the station and use its city
      const selectedStation = stations.find(s => s.stationPublicId === stationId);
      if (selectedStation) {
        locationParam = selectedStation.city.toLowerCase().replace(/\s+/g, '-');
      }
    }

    // Create new URLSearchParams object
    const params = new URLSearchParams(searchParams.toString());
    
    // Update or add location parameter
    if (locationParam) {
      params.set("location", locationParam);
    }
    
    // Remove lat/lon if they exist (since we're using location name now)
    params.delete("lat");
    params.delete("lon");
    
    // Update the URL without refreshing the page
    router.push(`${pathname}?${params.toString()}`, { scroll: false });
  }, [router, pathname, searchParams, stations, stationGroup]);

  const sortedStations = [...stations].sort((a, b) => {
    if (nearestStationId) {
      if (a.stationPublicId === nearestStationId) return -1;
      if (b.stationPublicId === nearestStationId) return 1;
    }

    if (a.state < b.state) return -1;
    if (a.state > b.state) return 1;
    if (a.city < b.city) return -1;
    if (a.city > b.city) return 1;
    return 0;
  });

  const nearestStation = nearestStationId
    ? stations.find((s) => s.stationPublicId === nearestStationId)
    : null;

  const handleStationSelect = (stationId: string) => {
    onStationSelect(stationId);
    updateUrlWithStation(stationId);
    setIsPopoverOpen(false);
  };

  const handleDetectNearest = () => {
    if (onDetectNearest) {
      onDetectNearest();
      // After detecting nearest, we'll update the URL via the parent component
      // The parent will call updateUrlWithStation after the nearest station is found
    }
    setIsPopoverOpen(false);
  };

  return (
    <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            "group flex min-h-11 items-start gap-1 text-left transition-opacity hover:opacity-90 focus-visible:rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring cursor-pointer select-none",
            triggerClassName
          )}
          aria-label={t("stationSelector.ariaLabel")}
        >
          <div className="flex flex-col">
            {/* Demoted affordance label — small, muted, with chevron */}
            <div className="flex items-center gap-1 mb-0.5">
              <p className="text-sm md:text-xs font-medium tracking-wide text-light uppercase">
                {t("stationSelector.eyebrow")}
              </p>
              <ChevronDown
                strokeWidth={3}
                className="w-3 h-3 text-light group-hover:text-white/80 transition-colors shrink-0"
              />
            </div>

            {/* Primary identity — city name dominates */}
            <div className="flex items-center gap-2">
              <p className={cn("font-bold leading-tight tracking-tight text-light text-2xl md:text-5xl", locationClassName ?? "text-2xl md:text-3xl")}>
                {station.city}
                {station.state && typeof station.state === "string" && station.state.trim() !== ""
                  ? ", " + station.state
                  : ""}
              </p>
            </div>

            {showAddress && (
              <p className="mt-0.5 text-[0.625rem] md:text-sm text-light">{station.address}</p>
            )}
          </div>
        </button>
      </PopoverTrigger>

      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          className={cn(
            "z-50 w-[calc(100vw-4rem)] sm:w-96 p-0 rounded-md border bg-popover text-popover-foreground shadow-xl outline-none",
            "data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
            "data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95",
            "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2"
          )}
          align="start"
          sideOffset={8}
          alignOffset={0}
        >
          <div className="max-h-[40vh] overflow-y-auto">

            {/* Detect nearest button */}
            {onDetectNearest && (
              <button
                type="button"
                onClick={handleDetectNearest}
                disabled={isLocating}
                className={cn(
                  "w-full flex min-h-11 items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring cursor-pointer",
                  "border-b border-border",
                  isLocating && "opacity-50 cursor-not-allowed"
                )}
              >
                {isLocating ? (
                  <Loader2 className="w-5 h-5 text-foreground animate-spin shrink-0" />
                ) : (
                  <LocateFixed className="w-5 h-5 text-foreground shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm text-foreground">
                    {isLocating ? t("stationSelector.detectingLocation") : t("stationSelector.findNearestStation")}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {isLocating ? t("stationSelector.pleaseWait") : t("stationSelector.useCurrentLocation")}
                  </p>
                </div>
              </button>
            )}

            {/* Nearest station — visually elevated */}
            {nearestStation && (
              <>
                <button
                  type="button"
                  onClick={() => handleStationSelect(nearestStation.stationPublicId)}
                  className={cn(
                    "w-full flex min-h-11 items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring cursor-pointer",
                    "border-b border-border",
                    selectedStationId === nearestStation.stationPublicId && "bg-accent/50"
                  )}
                >
                  <Home className="w-5 h-5 text-foreground shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-sm text-foreground">{nearestStation.stationName}</p>
                      {/* Nearest badge */}
                      <span className="inline-flex items-center rounded-full bg-blue-500/15 px-2 py-0.5 text-[0.625rem] font-semibold text-blue-500 ring-1 ring-blue-500/20">
                        {t("stationSelector.nearest")}
                      </span>
                      {selectedStationId === nearestStation.stationPublicId && (
                        <Check className="w-4 h-4 text-foreground shrink-0 ml-auto" />
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {nearestStation.state &&
                      typeof nearestStation.state === "string" &&
                      nearestStation.state.trim() !== ""
                        ? nearestStation.state + ", "
                        : ""}
                      {nearestStation.city}
                    </p>
                  </div>
                </button>

                {sortedStations.filter((s) => s.stationPublicId !== nearestStationId).length > 0 && (
                  <div className="px-4 py-2 border-b border-border">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      {t("stationSelector.allStations")}
                    </p>
                  </div>
                )}
              </>
            )}

            {/* All other stations */}
            {sortedStations
              .filter((s) => s.stationPublicId !== nearestStationId)
              .map((s) => (
                <button
                  type="button"
                  key={s.stationPublicId}
                  onClick={() => handleStationSelect(s.stationPublicId)}
                  className={cn(
                    "w-full flex min-h-11 items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring cursor-pointer",
                    selectedStationId === s.stationPublicId && "bg-accent/50"
                  )}
                >
                  <div className="w-5 h-5 shrink-0 flex items-center justify-center">
                    {selectedStationId === s.stationPublicId && (
                      <Check className="w-4 h-4 text-foreground" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm text-foreground">{s.stationName}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {s.state && typeof s.state === "string" && s.state.trim() !== ""
                        ? s.state + ", "
                        : ""}
                      {s.city}
                    </p>
                  </div>
                </button>
              ))}
          </div>
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </Popover>
  );
};
