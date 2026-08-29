"use client";

import React, { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Lock, User, ArrowRight, ShieldCheck, AlertCircle, Loader2, ArrowLeft } from "lucide-react";

export default function PortalLoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const res = await fetch("/api/portal/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        setError(data.message || "Invalid credentials. Please try again.");
        setIsLoading(false);
        return;
      }

      window.location.href = "/portal";
    } catch (err) {
      console.error("Login error:", err);
      setError("An unexpected error occurred. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0d] flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden">
      {/* Background ambient glowing gradients */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-orange-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 left-1/3 w-80 h-80 bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Back to Public Dashboard */}
      <div className="absolute top-6 left-6 z-20">
        <Link
          href="/weather"
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-zinc-900/80 hover:bg-zinc-800 border border-zinc-800 text-xs font-medium text-zinc-300 hover:text-white transition"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Public Weather</span>
        </Link>
      </div>

      <div className="w-full max-w-md z-10">
        {/* Card Container */}
        <div className="bg-[#121217]/90 backdrop-blur-2xl border border-zinc-800/90 rounded-3xl p-6 sm:p-8 shadow-2xl shadow-black/80 space-y-6">
          {/* Header & Emblem */}
          <div className="flex flex-col items-center text-center">
            <div className="w-16 h-16 rounded-2xl border border-orange-500/30 bg-orange-500/10 flex items-center justify-center p-3 mb-4 shadow-lg shadow-orange-500/10">
              <Image
                src="/icons/logo.png"
                alt="Kloudtrack Emblem"
                width={48}
                height={48}
                className="object-contain"
              />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Kloudtrack<span className="text-orange-500">.</span> Data Vault
            </h1>
            <p className="text-xs text-zinc-400 mt-1 max-w-xs">
              Secure export terminal for historical telemetry, processed physical metrics, and PINN nowcasts.
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="flex items-center gap-2.5 p-3.5 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="text-xs font-mono font-semibold uppercase text-zinc-300 block mb-1.5">
                Username
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-zinc-500">
                  <User className="w-4 h-4" />
                </div>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  placeholder="admin"
                  className="w-full h-11 pl-10 pr-4 rounded-xl bg-zinc-900/90 border border-zinc-750 text-white text-sm placeholder-zinc-500 focus:outline-none focus:border-orange-500 focus:ring-1 focus:ring-orange-500 transition"
                />
              </div>
            </div>

            <div>
              <label className="text-xs font-mono font-semibold uppercase text-zinc-300 block mb-1.5">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-zinc-500">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  placeholder="••••••••••••"
                  className="w-full h-11 pl-10 pr-4 rounded-xl bg-zinc-900/90 border border-zinc-750 text-white text-sm placeholder-zinc-500 focus:outline-none focus:border-orange-500 focus:ring-1 focus:ring-orange-500 transition"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full h-12 rounded-xl font-bold text-sm bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-400 hover:to-amber-400 text-black shadow-lg shadow-orange-500/20 hover:shadow-orange-500/30 flex items-center justify-center gap-2 transition transform active:scale-[0.98] cursor-pointer disabled:opacity-50 mt-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-black" />
                  <span>Authenticating...</span>
                </>
              ) : (
                <>
                  <span>Unlock Data Vault</span>
                  <ArrowRight className="w-4 h-4 text-black" />
                </>
              )}
            </button>
          </form>

          {/* Footer note */}
          <div className="pt-2 border-t border-zinc-800/80 text-center">
            <p className="text-[11px] text-zinc-400 flex items-center justify-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-orange-400/80" />
              <span>Signed session cookie • 24-hour authorization</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
