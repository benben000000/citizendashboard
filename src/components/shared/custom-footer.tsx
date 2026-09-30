"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";

export default function CustomFooter() {
  const [open, setOpen] = useState(false);

  return (
    <footer className="border-t border-border/50 bg-muted/30">
      <div className="max-w-360 mx-auto px-5 md:px-10 py-6 md:py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-foreground/70">
          <div className="flex flex-col md:flex-row items-center gap-2 text-center md:text-left">
            <p>© {new Date().getFullYear()} Kloudtech Corp. All rights reserved.</p>
            <span className="hidden md:inline text-border">•</span>
            <p className="text-muted-foreground">Powered by KloudTrack PINN-LNN Continuous-Time Engine</p>
          </div>

          <div className="flex items-center gap-4">
            <Link
              href="/portal"
              className="text-primary/90 hover:text-primary hover:underline font-medium transition-colors cursor-pointer"
            >
              Data Vault Export
            </Link>
            <span className="text-border">•</span>
            <Dialog open={open} onOpenChange={setOpen}>
              <DialogTrigger asChild>
                <button
                  type="button"
                  className="text-primary/90 hover:text-primary hover:underline font-medium transition-colors cursor-pointer"
                >
                  Data Sources & Attributions
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-xl max-h-[85vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle className="text-base font-semibold">
                    Data Sources, Scientific Attributions & Fair Use
                  </DialogTitle>
                  <DialogDescription className="text-xs text-muted-foreground pt-1">
                    Disclosure of technological provenance, academic attributions, and commercial licensing.
                  </DialogDescription>
                </DialogHeader>

                <div className="space-y-3.5 text-xs text-foreground/90 pt-2">
                  <div className="p-3 rounded-lg border border-border bg-card/40 space-y-1">
                    <p className="font-semibold text-foreground">
                      Liquid Neural Network (LNN) AI Architecture
                    </p>
                    <p className="text-muted-foreground leading-relaxed">
                      Continuous-time Liquid Neural Network (LNN) and Closed-form Continuous-time (CfC) differential equation principles are based on research by <strong>Hasani et al. (MIT CSAIL, 2021/2022)</strong>. All neural weight matrices and ODE kernels are proprietary Kloudtech assets.
                    </p>
                  </div>

                  <div className="p-3 rounded-lg border border-border bg-card/40 space-y-1">
                    <p className="font-semibold text-foreground">
                      External Data Sources — Not Currently In Use
                    </p>
                    <p className="text-muted-foreground leading-relaxed">
                      No third-party satellite or radar feed contributes to any value on this site. Adapters exist
                      for Himawari-9 satellite imagery, Doppler radar (RainViewer), and PAGASA numerical weather
                      prediction, but all are recorded as <strong>UNKNOWN_BLOCKED</strong> in the project&apos;s
                      external source registry (<code>prediction-model/data/external_source_registry.json</code>)
                      pending a commercial licence review. They currently return no data. Public API access does
                      not by itself constitute commercial or model-training rights.
                    </p>
                  </div>

                  <div className="p-3 rounded-lg border border-border bg-card/40 space-y-1">
                    <p className="font-semibold text-foreground">
                      Physical Ground Telemetry &amp; IoT Hardware
                    </p>
                    <p className="text-muted-foreground leading-relaxed">
                      Sub-second telemetry streams are produced by 23 physical Automated Weather Stations (AWS) and Water Level Monitoring Stations (WLMS) owned and operated by <strong>Kloudtech Inc.</strong> across Central Luzon and Bataan.
                    </p>
                  </div>

                  <div className="p-3 rounded-lg border border-amber-500/40 bg-amber-500/5 text-[11px] text-foreground/90 space-y-1">
                    <p className="font-semibold text-foreground">Model Status &amp; Limitations</p>
                    <p className="leading-relaxed">
                      The forecasting engine is a <strong>research prototype</strong> and is marked{" "}
                      <code>not_for_life_safety</code>. It is provided for monitoring, planning and research
                      only. Do not use it for evacuation, life-safety decisions, or emergency action.
                      River-stage output is a beta capability and is not approved for flood-warning use.
                      Weather prediction intervals are unavailable — probabilities are calibrated point
                      estimates without uncertainty bands, and most variables fall back to the last valid
                      observation. Where no measurement exists for a station, the interface shows an explicit
                      unavailable state rather than an estimate. Always follow PAGASA and your local government
                      unit for official warnings.
                    </p>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
            <p>Made in the Philippines</p>
          </div>
        </div>
      </div>
    </footer>
  );
}