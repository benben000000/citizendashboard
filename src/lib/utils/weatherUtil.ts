export const getWindDirectionLabel = (value: number | null | undefined) => {
  if (value === null || value === undefined) {
    return { direction: "--", compass: "--" };
  }

  const directions = [
    { name: "N", min: 337.5, max: 360 },
    { name: "N", min: 0, max: 22.5 },
    { name: "NE", min: 22.5, max: 67.5 },
    { name: "E", min: 67.5, max: 112.5 },
    { name: "SE", min: 112.5, max: 157.5 },
    { name: "S", min: 157.5, max: 202.5 },
    { name: "SW", min: 202.5, max: 247.5 },
    { name: "W", min: 247.5, max: 292.5 },
    { name: "NW", min: 292.5, max: 337.5 },
  ];

  const direction = directions.find((d) => value >= d.min && value < d.max);
  return {
    direction: direction?.name || "--",
    compass: `${Math.round(value)}°`,
  };
};