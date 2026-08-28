"use client";

import { useEffect, useState } from "react";

const getOverlayByTime = () => {
  const hour = new Date().getHours();

  if (hour >= 5 && hour < 8)
    return "from-[#182D35]/50 via-[#2C5364]/50 via-[#3F6C82]/50 via-[#5A9CB5]/50 via-[#7BC3E3]/50 to-transparent/50";

  if (hour >= 8 && hour < 12)
    return "from-[#6BA3D6]/50 via-[#8BB8E8]/50 via-[#9EBFE4]/50 via-[#A9C7E6]/50 via-[#B0C9E0]/50 to-transparent/50";

  if (hour >= 12 && hour < 17)
    return "from-[#57A7D0]/50 via-[#7BC3E3]/50 via-[#89CDE9]/50 via-[#92D6ED]/50 via-[#98DBF0]/50 to-transparent/50";

  if (hour >= 17 && hour < 19)
    return "from-[#C85A1E]/50 via-[#D4782A]/50 via-[#D98533]/50 via-[#DD9038]/50 via-[#E09640]/50 to-transparent/50";

  return "from-[#0F2027]/50 via-[#2C5364]/50 via-[#3B5561]/50 via-[#4A6570]/50 via-[#566F7A]/50 to-transparent/50";
};

export function TimeOverlay() {
  const [overlayClass, setOverlayClass] = useState<string | null>(null);

  useEffect(() => {
    const update = () => {
      const next = getOverlayByTime();
      setOverlayClass(prev => (prev === next ? prev : next));
    };

    update();
    const interval = setInterval(update, 60_000);
    return () => clearInterval(interval);
  }, []);

  if (!overlayClass) return null;

  return (
    <div
      className={`fixed inset-0 pointer-events-none bg-linear-to-b bg-[linear-gradient(to_right,#4FA3D1,#62B1DA,#74C0E3,#A8D8F0)] transition-colors duration-2000 z-0`}
    />
  );
}
