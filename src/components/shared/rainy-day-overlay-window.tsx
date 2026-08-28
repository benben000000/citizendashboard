import type { CSSProperties } from "react";

type RainIntensity = "light" | "steady";

type RainyDayOverlayProps = {
  intensity?: RainIntensity;
};

type RainDropPreset = {
  id: string;
  kind: "static" | "falling";
  density?: "desktop";
  left: number;
  top: number;
  width: number;
  height: number;
  opacity: number;
  duration: number;
  delay: number;
  drift: number;
  travel: number;
  trailLength: number;
  trailOpacity: number;
};

const LIGHT_DROPS = [
  { id: "l1",  kind: "falling", left: 8,  top: -8,  width: 11,  height: 13,  opacity: 0.38, duration: 18, delay: -9,  drift: 8,   travel: 122, trailLength: 52,  trailOpacity: 0.64 },
  { id: "l2",  kind: "static",  left: 18, top: 18,  width: 8,   height: 8,   opacity: 0.3,  duration: 24, delay: -12, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l3",  kind: "falling", left: 27, top: 34,  width: 9,   height: 11,  opacity: 0.42, duration: 24, delay: -18, drift: -9,  travel: 76,  trailLength: 44,  trailOpacity: 0.7 },
  { id: "l4",  kind: "static",  left: 40, top: 42,  width: 10,  height: 10,  opacity: 0.34, duration: 22, delay: -6,  drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l5",  kind: "falling", left: 51, top: -12, width: 12,  height: 14,  opacity: 0.4,  duration: 21, delay: -3,  drift: 13,  travel: 128, trailLength: 56,  trailOpacity: 0.66 },
  { id: "l6",  kind: "static",  left: 63, top: 31,  width: 8,   height: 8,   opacity: 0.38, duration: 27, delay: -20, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l7",  kind: "falling", left: 74, top: 48,  width: 10,  height: 12,  opacity: 0.43, duration: 28, delay: -15, drift: -12, travel: 62,  trailLength: 48,  trailOpacity: 0.72 },
  { id: "l8",  kind: "static",  left: 87, top: 56,  width: 13,  height: 13,  opacity: 0.31, duration: 26, delay: -8,  drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l9",  kind: "falling", left: 94, top: -4,  width: 10,  height: 12,  opacity: 0.36, duration: 19, delay: -14, drift: -7,  travel: 116, trailLength: 48,  trailOpacity: 0.62 },
  { id: "l10", kind: "static",  left: 12, top: 72,  width: 9,   height: 9,   opacity: 0.36, duration: 30, delay: -23, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l11", kind: "falling", left: 35, top: 58,  width: 8,   height: 10,  opacity: 0.35, duration: 23, delay: -11, drift: 8,   travel: 48,  trailLength: 40,  trailOpacity: 0.6 },
  { id: "l12", kind: "static",  left: 70, top: 78,  width: 6,   height: 6,   opacity: 0.3,  duration: 28, delay: -17, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l13", kind: "falling", density: "desktop", left: 4,  top: 34, width: 13,  height: 16,  opacity: 0.4,  duration: 26, delay: -22, drift: 14,  travel: 78,  trailLength: 64,  trailOpacity: 0.68 },
  { id: "l14", kind: "static",  density: "desktop", left: 22, top: 64, width: 9,   height: 9,   opacity: 0.34, duration: 31, delay: -4,  drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l15", kind: "falling", density: "desktop", left: 46, top: 42, width: 14,  height: 17,  opacity: 0.44, duration: 32, delay: -27, drift: -14, travel: 64,  trailLength: 68,  trailOpacity: 0.74 },
  { id: "l16", kind: "static",  density: "desktop", left: 58, top: 7,  width: 11,  height: 11,  opacity: 0.29, duration: 29, delay: -19, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "l17", kind: "falling", density: "desktop", left: 82, top: 26, width: 12,  height: 14,  opacity: 0.38, duration: 25, delay: -7,  drift: 12,  travel: 86,  trailLength: 56,  trailOpacity: 0.65 },
  { id: "l18", kind: "static",  density: "desktop", left: 96, top: 38, width: 10,  height: 10,  opacity: 0.32, duration: 33, delay: -25, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
] satisfies RainDropPreset[];

const STEADY_EXTRA_DROPS = [
  { id: "s1", kind: "falling", left: 15, top: -15, width: 12, height: 14, opacity: 0.5,  duration: 17, delay: -5,  drift: -12, travel: 132, trailLength: 56,  trailOpacity: 0.82 },
  { id: "s2", kind: "falling", left: 31, top: 44,  width: 10, height: 12, opacity: 0.48, duration: 22, delay: -16, drift: 14,  travel: 64,  trailLength: 48,  trailOpacity: 0.78 },
  { id: "s3", kind: "static",  left: 56, top: 62,  width: 11, height: 11, opacity: 0.42, duration: 25, delay: -10, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
  { id: "s4", kind: "falling", left: 67, top: 30,  width: 13, height: 16, opacity: 0.52, duration: 19, delay: -13, drift: -13, travel: 82,  trailLength: 64,  trailOpacity: 0.86 },
  { id: "s5", kind: "falling", density: "desktop", left: 2,  top: 8,  width: 14, height: 17, opacity: 0.46, duration: 21, delay: -18, drift: 10,  travel: 112, trailLength: 68,  trailOpacity: 0.78 },
  { id: "s6", kind: "falling", density: "desktop", left: 79, top: 44, width: 12, height: 14, opacity: 0.5,  duration: 27, delay: -24, drift: -20, travel: 82,  trailLength: 56,  trailOpacity: 0.84 },
  { id: "s7", kind: "static",  density: "desktop", left: 91, top: 15, width: 12, height: 12, opacity: 0.4,  duration: 28, delay: -21, drift: 0,   travel: 0,   trailLength: 0,   trailOpacity: 0 },
] satisfies RainDropPreset[];

const RAIN_PRESETS = {
  light: LIGHT_DROPS,
  steady: [...LIGHT_DROPS, ...STEADY_EXTRA_DROPS],
} satisfies Record<RainIntensity, RainDropPreset[]>;

export default function RainyDayOverlayWindow({ intensity = "light" }: RainyDayOverlayProps) {
  const overlayClassName = `rainy-day-overlay rainy-day-overlay--droplets rainy-day-overlay--${intensity}`;

  return (
    <div className={overlayClassName} aria-hidden="true">
      <div className="rainy-day-overlay__veil" />

      {RAIN_PRESETS[intensity].map((drop) => {
        const desktopClass = drop.density === "desktop" ? "--desktop" : "";
        const sharedStyle = {
          "--drop-left": `${drop.left}%`,
          "--drop-top": `${drop.top}%`,
          "--drop-width": `${drop.width}px`,
          "--drop-height": `${drop.height}px`,
          "--drop-opacity": drop.opacity,
          "--drop-duration": `${drop.duration}s`,
          "--drop-delay": `${drop.delay}s`,
          "--drop-drift": `${drop.drift}px`,
          "--drop-travel": `${drop.travel}vh`,
          "--drop-trail-length": `${drop.trailLength}px`,
          "--drop-trail-opacity": drop.trailOpacity,
        } as CSSProperties;

        if (drop.kind === "static") {
          return (
            <span
              key={drop.id}
              className={`rainy-day-overlay__drop rainy-day-overlay__drop--static${desktopClass ? ` rainy-day-overlay__drop${desktopClass}` : ""}`}
              style={sharedStyle}
            />
          );
        }

        // Falling drop: group wrapper owns position + animation,
        // trail and drop are children so the trail always follows
        return (
          <span
            key={drop.id}
            className={`rainy-day-overlay__drop-group${desktopClass ? ` rainy-day-overlay__drop-group${desktopClass}` : ""}`}
            style={sharedStyle}
          >
            <span className="rainy-day-overlay__trail" />
            <span className="rainy-day-overlay__drop rainy-day-overlay__drop--falling" />
          </span>
        );
      })}
    </div>
  );
}