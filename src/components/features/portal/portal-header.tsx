"use client";

import React, { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { LogOut, Database, ShieldCheck, ArrowLeft, Loader2 } from "lucide-react";

interface PortalHeaderProps {
  username?: string;
}

export const PortalHeader: React.FC<PortalHeaderProps> = ({ username = "admin" }) => {
  const router = useRouter();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = async () => {
    try {
      setIsLoggingOut(true);
      await fetch("/api/portal/auth/logout", { method: "POST" });
      router.push("/portal/login");
      router.refresh();
    } catch (e) {
      console.error("Logout failed:", e);
      setIsLoggingOut(false);
    }
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-orange-500/20 bg-[#0b0b0e]/90 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo & Portal Title */}
        <div className="flex items-center gap-3 sm:gap-4">
          <Link href="/weather" className="group flex items-center gap-2.5">
            <div className="relative w-9 h-9 rounded-xl overflow-hidden border border-orange-500/30 bg-orange-500/10 flex items-center justify-center p-1.5 transition group-hover:border-orange-400">
              <Image
                src="/icons/logo.png"
                alt="Kloudtrack Emblem"
                width={32}
                height={32}
                className="object-contain"
              />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base sm:text-lg tracking-tight text-white group-hover:text-orange-400 transition">
                  Kloudtrack<span className="text-orange-500">.</span>
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold uppercase tracking-wider bg-orange-500/20 text-orange-400 border border-orange-500/30">
                  Data Vault
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 hidden sm:block">
                Telemetry & Prediction Export Hub
              </p>
            </div>
          </Link>
        </div>

        {/* User Session Info & Navigation Actions */}
        <div className="flex items-center gap-3">
          <Link
            href="/weather"
            className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-zinc-300 hover:text-white bg-zinc-900/60 hover:bg-zinc-800 border border-zinc-800 transition"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Public Dashboard</span>
          </Link>

          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-900/80 border border-zinc-800 text-xs text-zinc-300">
            <ShieldCheck className="w-3.5 h-3.5 text-orange-400" />
            <span className="font-mono text-zinc-400">{username}</span>
          </div>

          <button
            onClick={handleLogout}
            disabled={isLoggingOut}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-orange-200 bg-orange-500/10 hover:bg-orange-500/20 border border-orange-500/30 hover:border-orange-500/50 transition cursor-pointer disabled:opacity-50"
            title="Log out of Data Vault"
          >
            {isLoggingOut ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <LogOut className="w-3.5 h-3.5 text-orange-400" />
            )}
            <span>Logout</span>
          </button>
        </div>
      </div>
    </header>
  );
};
