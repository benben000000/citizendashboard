"use client";

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/contexts/theme-context";

export function ThemeToggle() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-2 border-primary bg-secondary">
      {isDark ? (
        <Moon className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={2} />
      ) : (
        <Sun className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={2} />
      )}
      <span className="text-xs font-medium text-muted-foreground">{isDark ? "PH Night" : "PH Day"}</span>
    </div>
  );
}
