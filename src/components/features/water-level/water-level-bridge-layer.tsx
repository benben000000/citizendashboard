import type { ReactElement } from "react";

export const BRIDGE_REFERENCE_COLOR = "#374151";

interface ChartOffset {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface AxisWithScale {
  scale?: (value: number) => number;
}

interface WaterLevelBridgeLayerProps {
  referenceValue: number;
  referenceColor?: string;
  yAxisId?: number | string;
  offset?: ChartOffset;
  yAxisMap?: Record<string | number, AxisWithScale>;
  deckThicknessCm?: number;
  railingHeightCm?: number;
  visualStyle?: "realistic" | "compact";
}

export default function WaterLevelBridgeLayer({
  referenceValue,
  referenceColor = BRIDGE_REFERENCE_COLOR,
  yAxisId = 0,
  offset,
  yAxisMap,
  deckThicknessCm = 80,
  railingHeightCm = 110,
  visualStyle = "realistic",
}: WaterLevelBridgeLayerProps): ReactElement<SVGElement> | null {
  const yScale = yAxisMap?.[yAxisId]?.scale;
  if (!offset || typeof yScale !== "function") return null;

  const y = yScale(referenceValue);
  if (!Number.isFinite(y) || offset.width <= 0 || offset.height <= 0) return null;

  const yRange = yScale(1000) - yScale(0);
  const cmToPixels = Math.abs(yRange) / 1000;
  const deckHeightPx = Math.max(
    visualStyle === "compact" ? 4 : 3,
    deckThicknessCm * cmToPixels
  );
  const railingHeightPx = Math.max(
    visualStyle === "compact" ? 6 : 5,
    railingHeightCm * cmToPixels
  );
  const deckHeight = Math.min(deckHeightPx, offset.height * 0.15);
  const railingHeight = Math.min(railingHeightPx, offset.height * 0.12);
  const left = offset.left;
  const right = offset.left + offset.width;

  // The reference line and label always sit at the TRUE reference value
  // position — never clamped.
  const referenceLineY = y;

  // The bridge illustration is built relative to the true value position,
  // then shifted as a single rigid group if it would run outside the chart
  // bounds. This keeps posts/deck/railings in their correct relative
  // spacing — only their overall placement moves.
  const clampedBridgeY = Math.min(
    offset.top + offset.height - deckHeight - railingHeight,
    Math.max(offset.top + railingHeight + deckHeight, y)
  );
  const bridgeShiftY = clampedBridgeY - y;

  const inset = Math.min(16, (right - left) * 0.04);
  const startX = left + inset;
  const endX = right - inset;
  const bridgeWidth = endX - startX;

  // Geometry computed relative to the TRUE value position (y). The
  // enclosing <g transform="translate(0, bridgeShiftY)"> below moves the
  // whole illustration into bounds without disturbing this layout.
  const deckTopY = y - deckHeight / 2;
  const railTopY = deckTopY - railingHeight;

  const minPosts = visualStyle === "compact" ? 5 : 7;
  const maxPosts = visualStyle === "compact" ? 12 : 18;
  const postSpacingPx = visualStyle === "compact" ? 35 : 45;
  const postCount = Math.max(
    minPosts,
    Math.min(maxPosts, Math.floor(bridgeWidth / postSpacingPx))
  );
  const postSpacing = bridgeWidth / (postCount + 1);
  const beamCount = Math.max(3, Math.min(7, Math.floor(bridgeWidth / 80)));
  const beamSpacing = bridgeWidth / (beamCount + 1);

  return (
    <g pointerEvents="none" aria-hidden="true" style={{ color: referenceColor }}>
      <defs>
        <filter id="bridgeGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
          <feMerge>
            <feMergeNode in="offsetblur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <linearGradient id="bridgeDeckGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.15" />
          <stop offset="50%" stopColor="currentColor" stopOpacity="0.05" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.12" />
        </linearGradient>
        <linearGradient id="bridgeRailingGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.6" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.3" />
        </linearGradient>
      </defs>

      {/* Bridge illustration — shifted as a single rigid unit to stay
          within the chart bounds. Everything inside keeps its layout
          relative to the true reference value's y position. */}
      <g transform={`translate(0, ${bridgeShiftY})`}>
        <rect
          x={startX}
          y={deckTopY + deckHeight}
          width={bridgeWidth}
          height={Math.min(deckHeight * 2, offset.height * 0.1)}
          fill="currentColor"
          opacity={0.04}
        />

        <g opacity={0.6}>
          {[left, right].map((markerX) => (
            <g key={markerX}>
              <line
                x1={markerX}
                x2={markerX}
                y1={y - railingHeight * 0.4}
                y2={y + deckHeight * 0.4}
                stroke={referenceColor}
                strokeWidth={Math.max(1.5, Math.min(2, offset.width * 0.002))}
                strokeLinecap="round"
              />
              <circle
                cx={markerX}
                cy={y - railingHeight * 0.4}
                r={Math.max(2, Math.min(3, offset.width * 0.004))}
                fill={referenceColor}
                opacity={0.8}
              />
            </g>
          ))}
        </g>

        <g filter="drop-shadow(0 4px 8px rgba(15, 23, 42, 0.2))">
          <rect
            x={startX}
            y={deckTopY}
            width={bridgeWidth}
            height={deckHeight}
            rx={Math.min(3, deckHeight / 2)}
            fill="var(--background)"
            stroke="currentColor"
            strokeOpacity={0.6}
            strokeWidth={Math.max(1, Math.min(1.5, offset.width * 0.002))}
          />
          {deckHeight > 6 ? (
            <g stroke="currentColor" strokeOpacity={0.15} strokeWidth={0.5}>
              {[0.25, 0.5, 0.75].map((position) => (
                <line
                  key={position}
                  x1={startX + 2}
                  x2={endX - 2}
                  y1={deckTopY + deckHeight * position}
                  y2={deckTopY + deckHeight * position}
                />
              ))}
            </g>
          ) : null}
          {deckHeight > 5 ? (
            <g
              stroke="currentColor"
              strokeOpacity={0.3}
              strokeWidth={Math.max(0.8, deckHeight * 0.15)}
            >
              {Array.from({ length: beamCount }, (_, index) => {
                const beamX = startX + beamSpacing * (index + 1);

                return (
                  <line
                    key={beamX}
                    x1={beamX}
                    x2={beamX}
                    y1={deckTopY + deckHeight}
                    y2={deckTopY + deckHeight + Math.min(3, deckHeight * 0.3)}
                    strokeLinecap="round"
                  />
                );
              })}
            </g>
          ) : null}
          {Array.from({ length: postCount }, (_, index) => {
            const postX = startX + postSpacing * (index + 1);
            const postWidth = Math.max(1.5, Math.min(2.5, offset.width * 0.002));

            return (
              <g key={postX}>
                <line
                  x1={postX}
                  x2={postX}
                  y1={railTopY}
                  y2={deckTopY}
                  stroke="currentColor"
                  strokeWidth={postWidth}
                  strokeOpacity={0.7}
                  strokeLinecap="round"
                />
                {railingHeight > 8 ? (
                  <circle
                    cx={postX}
                    cy={railTopY}
                    r={Math.max(1.5, postWidth * 1.2)}
                    fill="currentColor"
                    opacity={0.8}
                  />
                ) : null}
              </g>
            );
          })}
          <line
            x1={startX}
            x2={endX}
            y1={railTopY}
            y2={railTopY}
            stroke="currentColor"
            strokeWidth={Math.max(1.2, Math.min(2, offset.width * 0.002))}
            strokeOpacity={0.85}
            strokeLinecap="round"
          />
          {railingHeight > 10 ? (
            <line
              x1={startX}
              x2={endX}
              y1={railTopY + railingHeight * 0.45}
              y2={railTopY + railingHeight * 0.45}
              stroke="currentColor"
              strokeWidth={Math.max(0.8, Math.min(1.2, offset.width * 0.0015))}
              strokeOpacity={0.5}
              strokeLinecap="round"
            />
          ) : null}
          {railingHeight > 12 ? (
            <line
              x1={startX}
              x2={endX}
              y1={railTopY + railingHeight * 0.75}
              y2={railTopY + railingHeight * 0.75}
              stroke="currentColor"
              strokeWidth={Math.max(0.7, Math.min(1, offset.width * 0.0012))}
              strokeOpacity={0.4}
              strokeLinecap="round"
            />
          ) : null}
          {railingHeight > 15 && postCount < 15 ? (
            <g stroke="currentColor" strokeOpacity={0.25} strokeWidth={0.6}>
              {Array.from({ length: postCount - 1 }, (_, index) => {
                const firstPostX = startX + postSpacing * (index + 1);
                const secondPostX = startX + postSpacing * (index + 2);
                const midY = railTopY + railingHeight * 0.6;

                return (
                  <g key={`${firstPostX}-${secondPostX}`}>
                    <line x1={firstPostX} y1={railTopY} x2={secondPostX} y2={midY} />
                    <line x1={firstPostX} y1={midY} x2={secondPostX} y2={railTopY} />
                  </g>
                );
              })}
            </g>
          ) : null}
        </g>
      </g>

      {/* Reference line — always aligned with the true reference value,
          independent of any bridge-illustration shift. */}
      <line
        x1={left}
        x2={right}
        y1={referenceLineY}
        y2={referenceLineY}
        stroke={referenceColor}
        strokeWidth={Math.max(2, Math.min(5, offset.height * 0.008))}
        strokeOpacity={0.05}
      />
      <line
        x1={left}
        x2={right}
        y1={referenceLineY}
        y2={referenceLineY}
        stroke={referenceColor}
        strokeWidth={Math.max(1.5, Math.min(2, offset.height * 0.003))}
        strokeDasharray="10 6"
        strokeOpacity={0.55}
        strokeLinecap="round"
      />

      {/* Label — pinned to the true reference value position. */}
      <g>
        <rect
          x={startX + bridgeWidth / 2 - 30}
          y={referenceLineY - 11}
          width={60}
          height={22}
          rx={4}
          fill="var(--background)"
          fillOpacity={0.9}
          stroke="var(--foreground)"
          strokeOpacity={0.3}
          strokeWidth={0.8}
        />
        <text
          x={startX + bridgeWidth / 2}
          y={referenceLineY}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="var(--foreground)"
          fontSize={10}
          fontWeight={600}
          opacity={0.95}
        >
          {referenceValue.toLocaleString()} cm
        </text>
      </g>
    </g>
  );
}
