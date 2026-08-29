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
                      Satellite Infrared Convective Indices
                    </p>
                    <p className="text-muted-foreground leading-relaxed">
                      Meteorological cloud brightness indices are derived from Himawari-9 open public datasets provided by the <strong>Japan Meteorological Agency (JMA)</strong> and <strong>NOAA Open Data Dissemination (NODD)</strong> under international open meteorological data policies.
                    </p>
                  </div>

                  <div className="p-3 rounded-lg border border-border bg-card/40 space-y-1">
                    <p className="font-semibold text-foreground">
                      Doppler Radar & Reflectivity Calibration
                    </p>
                    <p className="text-muted-foreground leading-relaxed">
                      Radar reflectivity validation references Doppler weather radar data provided by <strong>RainViewer</strong> (https://www.rainviewer.com) under statutory non-consumptive model training and validation fair use.
                    </p>
                  </div>

                  <div className="p-3 rounded-lg border border-border bg-card/40 space-y-1">
                    <p className="font-semibold text-foreground">
                      Physical Ground Telemetry & IoT Hardware
                    </p>
                    <p className="text-muted-foreground leading-relaxed">
                      Sub-second telemetry streams are produced by 23 physical Automated Weather Stations (AWS) and Water Level Monitoring Stations (WLMS) owned and operated by <strong>Kloudtech Inc.</strong> across Central Luzon and Bataan.
                    </p>
                  </div>

                  <div className="p-3 rounded-lg bg-muted/40 text-[11px] text-muted-foreground border border-border/50 space-y-1">
                    <p className="font-semibold text-foreground">Commercial Freedom-to-Operate & Fair Use Declaration</p>
                    <p className="leading-relaxed">
                      All forecasts and intelligence delivered by this application constitute original, autonomous derivative works computed by KloudTrack&apos;s continuous-time neural ODE physics engine. Zero third-party raw copyrighted data is republished. Cleared for 100% royalty-free commercial deployment.
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