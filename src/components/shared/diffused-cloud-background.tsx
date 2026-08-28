import type { CSSProperties } from "react";

type CloudSpec = {
  id: string;
  density?: "desktop";
  top: string;
  width: string;
  height: string;
  opacity: number;
  blur: string;
  delay: string;
  duration: string;
  driftStart: string;
  driftEnd: string;
  mobileX: string;
  tone: "soft" | "bright" | "shadow";
};

const CLOUDS: CloudSpec[] = [
  { id: "c1", top: "6%", width: "clamp(18rem, 58vw, 30rem)", height: "7.5rem", opacity: 0.34, blur: "0.08rem", delay: "-42s", duration: "170s", driftStart: "-32rem", driftEnd: "calc(100vw + 28rem)", mobileX: "-22vw", tone: "bright" },
  { id: "c2", top: "28%", width: "clamp(20rem, 68vw, 36rem)", height: "8.75rem", opacity: 0.26, blur: "0.11rem", delay: "-96s", duration: "220s", driftStart: "-38rem", driftEnd: "calc(100vw + 34rem)", mobileX: "18vw", tone: "shadow" },
  { id: "c3", top: "52%", width: "clamp(16rem, 52vw, 27rem)", height: "6.75rem", opacity: 0.3, blur: "0.09rem", delay: "-18s", duration: "185s", driftStart: "-28rem", driftEnd: "calc(100vw + 26rem)", mobileX: "-12vw", tone: "soft" },
  { id: "c4", top: "76%", width: "clamp(17rem, 56vw, 29rem)", height: "7.25rem", opacity: 0.28, blur: "0.1rem", delay: "-128s", duration: "205s", driftStart: "-30rem", driftEnd: "calc(100vw + 30rem)", mobileX: "28vw", tone: "soft" },
  { id: "c5", density: "desktop", top: "13%", width: "clamp(16rem, 24vw, 27rem)", height: "7.25rem", opacity: 0.32, blur: "0.1rem", delay: "-52s", duration: "165s", driftStart: "-27rem", driftEnd: "calc(100vw + 27rem)", mobileX: "0", tone: "soft" },
  { id: "c6", density: "desktop", top: "39%", width: "clamp(15rem, 22vw, 25rem)", height: "6.5rem", opacity: 0.4, blur: "0.07rem", delay: "-31s", duration: "145s", driftStart: "-25rem", driftEnd: "calc(100vw + 25rem)", mobileX: "0", tone: "bright" },
  { id: "c7", density: "desktop", top: "64%", width: "clamp(25rem, 40vw, 46rem)", height: "10.75rem", opacity: 0.24, blur: "0.14rem", delay: "-88s", duration: "210s", driftStart: "-46rem", driftEnd: "calc(100vw + 42rem)", mobileX: "0", tone: "shadow" },
  { id: "c8", density: "desktop", top: "82%", width: "clamp(17rem, 26vw, 30rem)", height: "7.5rem", opacity: 0.3, blur: "0.11rem", delay: "-22s", duration: "180s", driftStart: "-30rem", driftEnd: "calc(100vw + 30rem)", mobileX: "0", tone: "soft" },
];

export default function DiffusedCloudBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 z-0 overflow-hidden" aria-hidden>
      <svg className="cloud-filter-defs" width="0" height="0" focusable="false">
        <filter id="cloud-filter-base" x="-30%" y="-70%" width="160%" height="240%">
          <feTurbulence type="fractalNoise" baseFrequency="0.011" numOctaves="5" seed="8517" />
          <feDisplacementMap in="SourceGraphic" scale="120" />
        </filter>
        <filter id="cloud-filter-back" x="-34%" y="-90%" width="168%" height="280%">
          <feTurbulence type="fractalNoise" baseFrequency="0.011" numOctaves="3" seed="8517" />
          <feDisplacementMap in="SourceGraphic" scale="120" />
        </filter>
        <filter id="cloud-filter-mid" x="-34%" y="-90%" width="168%" height="280%">
          <feTurbulence type="fractalNoise" baseFrequency="0.011" numOctaves="3" seed="8517" />
          <feDisplacementMap in="SourceGraphic" scale="120" />
        </filter>
        <filter id="cloud-filter-front" x="-26%" y="-80%" width="152%" height="260%">
          <feTurbulence type="fractalNoise" baseFrequency="0.009" numOctaves="4" seed="8517" />
          <feDisplacementMap in="SourceGraphic" scale="50" />
        </filter>
      </svg>
      {CLOUDS.map((cloud) => (
        <div
          key={cloud.id}
          className={`diffused-cloud diffused-cloud--${cloud.tone}${
            cloud.density === "desktop" ? " diffused-cloud--desktop" : ""
          }`}
          style={{
            top: cloud.top,
            width: cloud.width,
            height: cloud.height,
            "--cloud-opacity": cloud.opacity,
            "--cloud-blur": cloud.blur,
            "--cloud-drift-start": cloud.driftStart,
            "--cloud-drift-end": cloud.driftEnd,
            "--cloud-mobile-x": cloud.mobileX,
            animationDelay: cloud.delay,
            animationDuration: cloud.duration,
          } as CSSProperties}
        >
          <span className="diffused-cloud__layer diffused-cloud__layer--base" />
          <span className="diffused-cloud__layer diffused-cloud__layer--back" />
          <span className="diffused-cloud__layer diffused-cloud__layer--mid" />
          <span className="diffused-cloud__layer diffused-cloud__layer--front" />
        </div>
      ))}
    </div>
  );
}
