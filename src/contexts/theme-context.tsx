"use client";
import React, { createContext, useContext, useState, useEffect } from "react";

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const PH_TIME_ZONE = "Asia/Manila";
const NIGHT_START_HOUR = 18;
const NIGHT_END_HOUR = 6;

function getPhilippineHour(date = new Date()) {
  const hour = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    hour12: false,
    timeZone: PH_TIME_ZONE,
  }).format(date);

  return Number(hour);
}

function getTimeBasedTheme(date = new Date()): Theme {
  const hour = getPhilippineHour(date);
  return hour >= NIGHT_START_HOUR || hour < NIGHT_END_HOUR ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => getTimeBasedTheme());
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setThemeState(getTimeBasedTheme());

    const intervalId = window.setInterval(() => {
      setThemeState(getTimeBasedTheme());
    }, 60_000);

    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;
    root.setAttribute("data-theme", theme);
    root.classList.toggle("dark", theme === "dark");
  }, [theme, mounted]);

  return <ThemeContext.Provider value={{ theme }}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
