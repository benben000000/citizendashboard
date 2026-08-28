"use client";

import { ArrowUpRight } from "lucide-react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";

const LEARN_MORE_URL = "https://homepage.kloudtechsea.com/";
const CONTACT_URL = "https://homepage.kloudtechsea.com/contact-us";
const STATION_BANNER_BG = "/images/banner.png";

export default function StationCtaBanner() {
  const t = useTranslations();

  return (
    <div className="mt-12 border-t border-border/50 bg-background/70 md:bg-background/20">
      <div className="max-w-360 mx-auto flex flex-col md:flex-row">

        {/* Left — Image */}
        <div className="w-full md:w-3/7 hidden md:block h-80 relative overflow-hidden">
          <div
            className="station-cta-banner-image absolute inset-0 bg-cover bg-center"
            style={{
              backgroundImage: `url(${STATION_BANNER_BG})`,
              backgroundPosition: "center 20%",
            }}
            aria-hidden="true"
          />
        </div>

        {/* Right — CTA */}
        <div className="w-full md:w-4/7 flex items-center px-6 md:px-10 py-8 md:py-10">
          <div className="flex flex-col gap-5">
            <div>
              <h2 className="text-2xl md:text-4xl font-bold text-light">
                {t("dashboard.ctaBanner.title")}
              </h2>
              <p className="mt-2 font-light text-xs md:text-lg text-light">
                {t("dashboard.ctaBanner.description")}
              </p>
            </div>

            <div className="flex flex-row sm:items-center gap-2">
              <Button
                className="border bg-white text-slate-800 border-main text-[0.625rem] md:text-sm h-8 md:h-12 px-4"
                variant="outline"
                size="lg"
                onClick={() => window.open(LEARN_MORE_URL, "_blank")}
              >
                {t("dashboard.ctaBanner.learnMore")} <ArrowUpRight className="ml-2" />
              </Button>

              <Button
                className="bg-main h-8 md:h-12 text-[0.625rem] md:text-sm px-4"
                variant="outline"
                size="lg"
                onClick={() => window.open(CONTACT_URL, "_blank")}
              >
                {t("dashboard.ctaBanner.contactUs")} <ArrowUpRight className="ml-2" />
              </Button>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
