"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { CloudSun, Waves, Search } from "lucide-react";

import { cn } from "@/lib/utils/cn";

import LanguageSelector from "./language-selector";

const NAV_ITEMS = [
  {
    href: "/weather",
    label: "Weather",
    icon: CloudSun,
  },
  {
    href: "/water-level",
    label: "Water level",
    icon: Waves,
  },
  {
    href: "/prediction",
    label: "Prediction",
    icon: Search,
  },
] as const;

export default function PublicDashboardTopBar() {
  const pathname = usePathname();

  return (
    <div className="flex w-full items-center gap-2 sm:justify-between">
      <Link href="/weather" className="inline-flex shrink-0 items-center">
        <img
          src="/favicon.ico"
          alt="Kloudtech logo"
          className="mr-1 h-5 w-5 md:mr-2 md:h-8 md:w-8"
        />
        <p className="inline-flex text-lg font-bold tracking-normal md:text-2xl">
          <span className="text-c_secondary">Kloud</span>
          <span className="text-main">tech</span>
        </p>
      </Link>

      <div className="flex min-w-0 flex-1 items-center justify-end gap-1.5 sm:flex-none sm:gap-2">
        <nav
          aria-label="Dashboard navigation"
          className="grid h-9 min-w-0 grid-cols-3 rounded-lg border border-slate-950/10 bg-white/65 p-1 shadow-sm backdrop-blur-md sm:h-10 sm:w-auto sm:flex-none"
        >
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive =
              pathname === item.href || pathname.startsWith(`${item.href}/`);

            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "inline-flex min-w-0 items-center justify-center gap-1.5 rounded-md px-2 text-xs font-semibold text-slate-600 transition-colors hover:bg-white/75 hover:text-slate-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-main/70 sm:gap-2 sm:px-3 sm:text-sm",
                  isActive && "bg-main text-slate-950 shadow-sm hover:bg-main"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                <span className="sr-only sm:not-sr-only sm:truncate">
                  {item.label}
                </span>
              </Link>
            );
          })}
        </nav>
        <LanguageSelector />
      </div>
    </div>
  );
}
