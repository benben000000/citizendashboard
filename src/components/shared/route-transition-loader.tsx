"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname, useSearchParams } from "next/navigation";

import DashboardLoadingScreen from "./dashboard-loading-screen";

const LOADER_TIMEOUT_MS = 2500;

type DashboardZone = "weather" | "water-level" | "prediction";

function getDashboardZone(pathname: string): DashboardZone | null {
  if (pathname === "/weather" || pathname.startsWith("/weather/")) {
    return "weather";
  }

  if (pathname === "/water-level" || pathname.startsWith("/water-level/")) {
    return "water-level";
  }

  if (pathname === "/prediction" || pathname.startsWith("/prediction/")) {
    return "prediction";
  }

  return null;
}

function getInternalAnchor(target: EventTarget | null): HTMLAnchorElement | null {
  if (!(target instanceof Element)) return null;
  return target.closest("a[href]");
}

function getInternalDestination(anchor: HTMLAnchorElement): URL | null {
  if (anchor.target && anchor.target !== "_self") return null;
  if (anchor.hasAttribute("download")) return null;

  const href = anchor.getAttribute("href");
  if (!href || href.startsWith("#") || href.startsWith("mailto:") || href.startsWith("tel:")) {
    return null;
  }

  const nextUrl = new URL(anchor.href);
  const currentUrl = new URL(window.location.href);

  if (
    nextUrl.origin === currentUrl.origin &&
    `${nextUrl.pathname}${nextUrl.search}` !== `${currentUrl.pathname}${currentUrl.search}`
  ) {
    return nextUrl;
  }

  return null;
}

export default function RouteTransitionLoader() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [isLoading, setIsLoading] = useState(false);
  const [destinationZone, setDestinationZone] = useState<DashboardZone | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }

      const anchor = getInternalAnchor(event.target);
      if (!anchor) return;

      const nextUrl = getInternalDestination(anchor);
      if (!nextUrl) return;

      const currentZone = getDashboardZone(window.location.pathname);
      const nextZone = getDashboardZone(nextUrl.pathname);

      if (!currentZone || !nextZone || currentZone === nextZone) return;

      setDestinationZone(nextZone);
      setIsLoading(true);

      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        setIsLoading(false);
      }, LOADER_TIMEOUT_MS);
    };

    document.addEventListener("click", handleClick, true);

    return () => {
      document.removeEventListener("click", handleClick, true);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  useEffect(() => {
    setIsLoading(false);
    setDestinationZone(null);

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, [pathname, searchParams]);

  return isLoading ? <DashboardLoadingScreen destination={destinationZone} /> : null;
}
