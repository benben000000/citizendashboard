import PublicDashboardTopBar from "@/components/shared/public-dashboard-top-bar";
import type { StationPublicInfo } from "@/types/telemetry";

interface Props {
  stations: StationPublicInfo[];
  selectedId: string;
  onSelect: (id: string) => void;
  onUseLocation?: () => void;
  isLocating?: boolean;
  locationError?: string | null;
  nearestStationId?: string | null;
}

const StationHeader = (_props: Props) => {
  return <PublicDashboardTopBar />;
};

export default StationHeader;
