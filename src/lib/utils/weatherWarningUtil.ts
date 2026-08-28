import {
  Footprints, Droplets, Clock, Tractor,
  ShoppingBag, TreePine, ThermometerSun, Wind,
  AlertTriangle, Baby, HeartPulse, Shirt,
  Home, CalendarX, Eye, ShieldAlert,
  BedDouble, GlassWater, Users, Phone,
  Umbrella, CloudRain, Route, Flashlight,
  Smartphone, MapPin, HandHelping, Waves,
} from "lucide-react";
import type { ComponentType } from "react";
import {
  DEFAULT_LOCALE,
  getWeatherActionCopy,
  getWeatherCommonCopy,
  getWeatherWarningCopy,
  type WeatherActionCopyKey,
  type WeatherWarningCopyKey,
  type Locale,
} from "@/lib/i18n/translations";

export interface WeatherForecastDetails {
  // Normalized details: only the numeric value (if any) and a human description.
  value: number | null;
  description: string;
}

export interface ReferenceWarning {
  color: string;
  term: string;
  warningLevel: string;
  suggestionAction?: string;
  whatCanYouDo?: WeatherActionItem[];
}

export interface WarningInfo {
  color: string | null;
  term: string;
}

type WeatherActionItem = { icon: ComponentType; text: string };
type WeatherActionReference = { icon: ComponentType; textKey: WeatherActionCopyKey };
type ReferenceThreshold = {
  min?: number;
  max?: number;
  color?: string;
  copyKey: WeatherWarningCopyKey;
  whatCanYouDo?: WeatherActionReference[];
};

// Thresholds for each type
export const REFERENCE_THRESHOLDS: Record<string, ReferenceThreshold[]> = {
  "Heat Index": [
    {
      min: 27,
      max: 32,
      color: "#FBF300",
      copyKey: "heatIndexCaution",
      whatCanYouDo: [
        { icon: GlassWater, textKey: "drinkWaterOften" },
        { icon: Home, textKey: "takeShadeBreaks" },
        { icon: Shirt, textKey: "wearLightClothing" },
        { icon: ThermometerSun, textKey: "avoidHottestHours" },
        { icon: Eye, textKey: "watchHeatStress" },
      ],
    },
    {
      min: 33,
      max: 41,
      color: "#F6CC47",
      copyKey: "heatIndexExtremeCaution",
      whatCanYouDo: [
        { icon: CalendarX, textKey: "limitOutdoorActivity" },
        { icon: Home, textKey: "stayShadedVentilated" },
        { icon: GlassWater, textKey: "drinkWaterFrequently" },
        { icon: ShieldAlert, textKey: "useSunProtection" },
        { icon: Eye, textKey: "stopAndCoolDown" },
      ],
    },
    {
      min: 42,
      max: 51,
      color: "#EC6F31",
      copyKey: "heatIndexDanger",
      whatCanYouDo: [
        { icon: Home, textKey: "avoidOutdoorExposure" },
        { icon: BedDouble, textKey: "stayCoolestIndoors" },
        { icon: GlassWater, textKey: "drinkWaterThroughoutDay" },
        { icon: Eye, textKey: "monitorHeatIllness" },
        { icon: Phone, textKey: "seekMedicalHelp" },
      ],
    },
    {
      min: 52,
      color: "#C13030",
      copyKey: "heatIndexExtremeDanger",
      whatCanYouDo: [
        { icon: BedDouble, textKey: "stayIndoorsPossible" },
        { icon: Home, textKey: "avoidOutdoorActivity" },
        { icon: GlassWater, textKey: "stayHydratedCool" },
        { icon: Eye, textKey: "watchHeatStroke" },
        { icon: Phone, textKey: "callEmergencyHelp" },
      ],
    },
  ],
  "Wind Speed": [
    { min: 39, max: 61, color: "#00CCFF", copyKey: "windTropicalDepression" },
    { min: 62, max: 88, color: "#FBF300", copyKey: "windTropicalStorm" },
    { min: 89, max: 117, color: "#FFA800", copyKey: "windSevereTropicalStorm" },
    { min: 118, max: 184, color: "#E63946", copyKey: "windTyphoon" },
    { min: 185, color: "#CB00CE", copyKey: "windSuperTyphoon" },
  ],
  Precipitation: [
    {
      min: 0.1,
      max: 2.5,
      color: "#B0E0E6",
      copyKey: "precipitationLightRain",
      whatCanYouDo: [
        { icon: Umbrella, textKey: "bringRainProtection" },
        { icon: Eye, textKey: "checkWeatherUpdates" },
        { icon: Footprints, textKey: "wearNonSlipFootwear" },
        { icon: ShoppingBag, textKey: "coverImportantItems" },
        { icon: Clock, textKey: "allowExtraTravelTime" },
      ],
    },
    {
      min: 2.5,
      max: 7.5,
      color: "#00BFFF",
      copyKey: "precipitationModerateRain",
      whatCanYouDo: [
        { icon: Footprints, textKey: "avoidSlipperyAreas" },
        { icon: Home, textKey: "secureOutdoorItems" },
        { icon: CloudRain, textKey: "monitorRainConditions" },
        { icon: Smartphone, textKey: "protectElectronics" },
        { icon: Route, textKey: "slowDownOnRoads" },
      ],
    },
    {
      min: 7.5,
      max: 15,
      color: "#FACC15",
      copyKey: "precipitationHeavyRain",
      whatCanYouDo: [
        { icon: Clock, textKey: "delayNonEssentialTravel" },
        { icon: Route, textKey: "avoidLowLyingRoads" },
        { icon: Flashlight, textKey: "prepareEmergencyLighting" },
        { icon: Eye, textKey: "watchForPoorVisibility" },
        { icon: Users, textKey: "monitorFamilyMembers" },
      ],
    },
    {
      min: 15,
      max: 30,
      color: "#FFA500",
      copyKey: "precipitationFloodRisk",
      whatCanYouDo: [
        { icon: Home, textKey: "moveValuablesHigher" },
        { icon: ShoppingBag, textKey: "prepareEmergencyBag" },
        { icon: Route, textKey: "avoidFloodProneRoutes" },
        { icon: AlertTriangle, textKey: "followLocalAdvisories" },
        { icon: MapPin, textKey: "checkEvacuationOptions" },
      ],
    },
    {
      min: 30,
      color: "#DC3545",
      copyKey: "precipitationSevereFlooding",
      whatCanYouDo: [
        { icon: Home, textKey: "stayIndoorsIfPossible" },
        { icon: Waves, textKey: "avoidFloodwaters" },
        { icon: HandHelping, textKey: "assistVulnerablePeople" },
        { icon: MapPin, textKey: "prepareForEvacuation" },
        { icon: Phone, textKey: "contactEmergencyServices" },
      ],
    },
  ],
  "UV Index": [
    { min: 3, max: 5, color: "#FFBC01", copyKey: "uvWearSunscreen" },
    { min: 6, max: 7, color: "#FF9000", copyKey: "uvSeekShade" },
    { min: 8, max: 10, color: "#F55023", copyKey: "uvAvoidSun" },
    { min: 11, color: "#9E47CC", copyKey: "uvStayInside" },
  ],
  Temperature: [
    { min: 0, max: 24, color: "#00BFFF", copyKey: "temperatureCool" },
    { min: 25, max: 29, color: "#7FFF00", copyKey: "temperatureWarm" },
    { min: 30, max: 34, color: "#FFBC01", copyKey: "temperatureGettingHot" },
    { min: 35, max: 39, color: "#FF9000", copyKey: "temperatureVeryHot" },
    { min: 40, color: "#E63946", copyKey: "temperatureDangerous" },
  ],
  Humidity: [
    { max: 30, color: "#D97706", copyKey: "humidityVeryDry" },
    { min: 31, max: 69, color: "#3B82F6", copyKey: "humidityComfortable" },
    { min: 70, max: 79, color: "#F59E0B", copyKey: "humidityHumid" },
    { min: 80, max: 100, color: "#DC2626", copyKey: "humidityVeryHumid" },
  ],
  Pressure: [
    { max: 979, color: "#1D4ED8", copyKey: "pressureStormLikely" },
    { min: 980, max: 990, color: "#3B82F6", copyKey: "pressureRainPossible" },
    { min: 991, max: 1030, color: "#60A5FA", copyKey: "pressureFairWeather" },
    { min: 1031, max: 1040, color: "#93C5FD", copyKey: "pressureClearCalm" },
    { min: 1041, color: "#BFDBFE", copyKey: "pressureVeryDryClear" },
  ],
  Light: [
    { max: 50, color: "#B0C4DE", copyKey: "lightVeryDim" },
    { min: 51, max: 200, color: "#ADD8E6", copyKey: "lightIndoor" },
    { min: 201, max: 500, color: "#FFFFE0", copyKey: "lightBrightIndoors" },
    { min: 501, max: 1000, color: "#FFFF99", copyKey: "lightVeryBright" },
    { min: 1001, max: 10000, color: "#FFFACD", copyKey: "lightCloudyDay" },
    { min: 10001, max: 25000, color: "#FFD700", copyKey: "lightBrightDay" },
    { min: 25001, max: 100000, color: "#FFA500", copyKey: "lightDirectSun" },
  ],
};

export const DEFAULT_WHAT_CAN_YOU_DO: WeatherActionReference[] = [
  { icon: GlassWater, textKey: "defaultDrinkWater" },
  { icon: Shirt, textKey: "defaultWearComfortableClothing" },
  { icon: Eye, textKey: "defaultCheckUpdates" },
  { icon: Home, textKey: "defaultStayCool" },
];

const resolveActionItems = (
  actions: WeatherActionReference[],
  locale: Locale = DEFAULT_LOCALE,
): WeatherActionItem[] => {
  return actions.map(({ icon, textKey }) => ({
    icon,
    text: getWeatherActionCopy(textKey, locale),
  }));
};

const resolveWarning = (
  threshold: ReferenceThreshold,
  locale: Locale = DEFAULT_LOCALE,
): ReferenceWarning => {
  const copy = getWeatherWarningCopy(threshold.copyKey, locale);

  return {
    color: threshold.color as string,
    term: copy.term,
    warningLevel: copy.warningLevel,
    suggestionAction: "suggestedAction" in copy ? copy.suggestedAction : undefined,
    whatCanYouDo: threshold.whatCanYouDo
      ? resolveActionItems(threshold.whatCanYouDo, locale)
      : undefined,
  };
};

export const getWhatCanYouDo = (
  whatCanYouDo?: WeatherActionItem[],
  locale: Locale = DEFAULT_LOCALE,
): WeatherActionItem[] => {
  return whatCanYouDo?.length ? whatCanYouDo : resolveActionItems(DEFAULT_WHAT_CAN_YOU_DO, locale);
};

export const getNormalWarningLabel = (
  locale: Locale = DEFAULT_LOCALE,
): string => getWeatherCommonCopy("normal", locale);


export function getReferenceWarning(
  type: string,
  value: number,
  applyWarning: boolean = true,
  locale: Locale = DEFAULT_LOCALE,
): ReferenceWarning | null {
  if (!applyWarning) {
    return null;
  }

  const typeThresholds = REFERENCE_THRESHOLDS[type];
  if (!typeThresholds) return null;

  // If thresholds are all integer bounds, round to nearest whole number to avoid gaps (e.g., 41.4 vs 41-42)
  const hasDecimalBounds = typeThresholds.some(
    (t) => (t.min != null && !Number.isInteger(t.min)) || (t.max != null && !Number.isInteger(t.max)),
  );
  const compareValue = hasDecimalBounds ? value : Math.round(value);

  for (const t of typeThresholds) {
    if (t.min !== undefined && t.max !== undefined) {
      if (compareValue >= t.min && compareValue <= t.max) {
        return resolveWarning(t, locale);
      }
    } else if (t.min !== undefined) {
      if (compareValue >= t.min) {
        return resolveWarning(t, locale);
      }
    } else if (t.max !== undefined) {
      if (compareValue <= t.max) {
      return resolveWarning(t, locale);
      }
    }
  }
  return null;
}

export const getWarningInfo = (
  type: string,
  value?: number | null,
  locale: Locale = DEFAULT_LOCALE,
): WarningInfo => {
  // Map type to util type
  const typeMap: Record<string, string> = {
    heatIndex: "Heat Index",
    temperature: "Temperature",
    windSpeed: "Wind Speed",
    uvIndex: "UV Index",
    precipitation: "Precipitation",
    humidity: "Humidity",
    pressure: "Pressure",
    light: "Light",
  };
  const refType = typeMap[type];
  if (!refType || value === null || value === undefined) {
    return { color: null, term: getWeatherCommonCopy("normal", locale) };
  }
  // Delegate rounding logic to getReferenceWarning so decimal-bounded types (e.g., precipitation) aren't rounded
  const ref = getReferenceWarning(refType, Number(value), true, locale);
  if (ref) return { color: ref.color as WarningInfo["color"], term: ref.term };
  return { color: null, term: getWeatherCommonCopy("normal", locale) };
};

export const getWarningStyles = (color: string | undefined): { bg: string; border: string; text: string } => {
  if (!color || color === "var(--foreground)") {
    return {
      bg: "rgba(128, 128, 128, 0.1)",
      border: "rgba(128, 128, 128, 0.3)",
      text: "rgba(128, 128, 128, 1)",
    };
  }

  const colorMap: Record<string, { bg: string; border: string; text: string }> = {
    // Heat Index — Caution
    "#FBF300": {
      bg: "var(--reference-yellow-bg)",
      border: "var(--reference-yellow-border)",
      text: "var(--reference-yellow-text)",
    },
    // // Heat Index — Extreme Caution
    // "#F6CC47": {
    //   bg: "var(--reference-amber-bg)",
    //   border: "var(--reference-amber-border)",
    //   text: "var(--reference-amber-text)",
    // },
    // // Heat Index — Danger
    // "#EC6F31": {
    //   bg: "var(--reference-orange-bg)",
    //   border: "var(--reference-orange-border)",
    //   text: "var(--reference-orange-text)",
    // },
    // // Heat Index — Extreme Danger
    // "#C13030": {
    //   bg: "var(--reference-red-bg)",
    //   border: "var(--reference-red-border)",
    //   text: "var(--reference-red-text)",
    // },
  };

  if (colorMap[color]) return colorMap[color];

  // Fallback for all other types (Wind Speed, UV, Precipitation, etc.)
  const hexToRgb = (hex: string): { r: number; g: number; b: number } | null => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16),
        }
      : null;
  };

  const rgb = hexToRgb(color);
  if (!rgb) {
    return {
      bg: "rgba(128, 128, 128, 0.1)",
      border: "rgba(128, 128, 128, 0.3)",
      text: "rgba(128, 128, 128, 1)",
    };
  }

  return {
    bg: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.1)`,
    border: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.3)`,
    text: `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 1)`,
  };
};

export const darkenColor = (color: string, amount: number = 0.2): string => {
  const factor = 1 - amount;

  // Handle rgb/rgba colors: rgb(255, 0, 0) or rgba(255, 0, 0, 0.3)
  // Always return a solid color (no transparency)
  if (color.startsWith("rgb")) {
    const match = /rgba?\(([^)]+)\)/.exec(color.replace(/\s+/g, ""));
    if (!match) return color;

    const parts = match[1].split(",");
    if (parts.length < 3) return color;

    const r = Number(parts[0]);
    const g = Number(parts[1]);
    const b = Number(parts[2]);

    if ([r, g, b].some((v) => Number.isNaN(v))) return color;

    const darkR = Math.max(0, Math.min(255, Math.floor(r * factor)));
    const darkG = Math.max(0, Math.min(255, Math.floor(g * factor)));
    const darkB = Math.max(0, Math.min(255, Math.floor(b * factor)));

    // Return opaque rgb, ignoring any original alpha
    return `rgb(${darkR}, ${darkG}, ${darkB})`;
  }

  // Handle hex colors: #rgb or #rrggbb
  const hex = color.startsWith("#") ? color.slice(1) : color;

  let normalized = hex;
  if (normalized.length === 3) {
    normalized = normalized
      .split("")
      .map((ch) => ch + ch)
      .join("");
  }

  if (normalized.length !== 6) {
    return color;
  }

  const num = parseInt(normalized, 16);
  const r = (num >> 16) & 0xff;
  const g = (num >> 8) & 0xff;
  const b = num & 0xff;

  const darkR = Math.max(0, Math.min(255, Math.floor(r * factor)));
  const darkG = Math.max(0, Math.min(255, Math.floor(g * factor)));
  const darkB = Math.max(0, Math.min(255, Math.floor(b * factor)));

  const darkNum = (darkR << 16) | (darkG << 8) | darkB;
  return `#${darkNum.toString(16).padStart(6, "0")}`;
};
