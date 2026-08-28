import { PARAMETERS } from "@/lib/constants/parameters";
import { WEATHER_REFERENCES } from "@/lib/constants/weather-references";
import { getWarningStyles } from "@/lib/utils/weatherWarningUtil";
import Link from "next/link";
import { Sparkles } from "lucide-react";
import { useTranslations } from "next-intl";

interface StationWeatherMetricTerminologyProps {
  selectedMetric: string;
}

const StationWeatherMetricTerminology = ({ selectedMetric }: StationWeatherMetricTerminologyProps) => {
  const t = useTranslations();
  const parameterName = PARAMETERS.find(param => param.key === selectedMetric)?.label || selectedMetric;   
  const references = parameterName ? WEATHER_REFERENCES[parameterName as keyof typeof WEATHER_REFERENCES] : undefined;
  const heading = t("terminology.understanding").replace("{metric}", parameterName);
  
  if (!references || references.length === 0) {
    return (
      <div className="mb-8">
          
        <div className="flex justify-between items-center  mb-4 ">
          <h2 className="border-l-4 border-l-main pl-2 text-lg md:text-xl font-semibold">
            {heading}
          </h2>
          <Link
            href="/terminologies"
            className="inline-flex items-center px-3 py-1 rounded-full text-xs md:text-sm font-medium bg-card/70 border border-border/30 hover:bg-card/90 hover:text-main transition-colors"
            aria-label={t("dashboard.metrics.guidesAriaLabel")}
          >
            <Sparkles className="h-4 w-4 mr-1 text-[#eab308]" />
            {t("dashboard.metrics.guides")}
          </Link>
        </div>
        
        <p>{t("terminology.noReferences")}</p>
      </div>
    );
  }
  
  return (
    <div className="">
      <div className="flex md:flex-row flex-col md:justify-between justify-start md:items-center  mb-4 ">
        <h2 className="border-l-4 border-l-main pl-2 text-lg md:text-xl font-semibold">
          {heading}
        </h2>
        <div className="md:mt-0 mt-2">
          <Link
            href="/terminologies"
            className="inline-flex items-center px-3 py-1 rounded-full text-xs md:text-sm font-medium bg-card/70 border border-border/30 hover:bg-card/90 hover:text-main transition-colors"
            aria-label={t("dashboard.metrics.guidesAriaLabel")}
          >
            <Sparkles className="h-4 w-4 mr-1 text-[#eab308]" />
            {t("dashboard.metrics.guides")}
          </Link>
        </div>
        
      </div>
      
      {/* Compact threshold bar */}
      {/* <div className="grid grid-cols-2 md:flex md:flex-row mb-4 bg-card">
        {references.map((ref, idx) => {
          const warningStyles = getWarningStyles(ref.color as string | undefined);
          const isLast = idx === references.length - 1;
          const isOddCount = references.length % 2 === 1;

          return (
            <div
              key={idx}
              className={[
                "font-bold text-center border md:flex-1 text-slate-600 ",
                isLast && isOddCount ? "col-span-2 md:col-span-1" : ""
              ].join(" ")}
              style={{
                backgroundColor: warningStyles.bg,
                borderColor: warningStyles.border,
              }}
            >
              {ref.threshold}
            </div>
          );
        })}
      </div> */}

      
      {/* Detailed reference cards */}
      <div className="flex flex-col gap-4">
        {references.map((ref, idx) => {
          const warningStyles = getWarningStyles(ref.color as string | undefined);
          return (
            <div key={idx} className="flex gap-4 items-start">
              {/* Threshold box */}
              <div className="bg-card shrink-0 hidden md:flex">
                <div
                  className="flex items-center text-slate-600 p-4 justify-center break-normal whitespace-normal min-w-52 max-w-52 font-bold text-center border"
                  style={{
                    backgroundColor: warningStyles.bg,
                    borderColor: warningStyles.border,
                   
                  }}
                >
                  {ref.threshold}
                </div>
              </div>

              {/* Term + Definition */}
              <div className="flex flex-col justify-start">
                <div className="font-semibold text-left text-light">
                  <div className="hidden md:block text-md ">
                    {ref.term}
                  </div>
                  <div className="block md:hidden mt-2">
                    {ref.term} 
                    <span className="ml-2 text-sm font-medium px-2 text-light py-1 rounded-full border" style={{ backgroundColor: warningStyles.bg, borderColor: warningStyles.border }}>
                      {ref.threshold}
                    </span>
                  </div>
                  
                </div>
                <p className="text-sm text-left mt-2 text-muted-foreground wrap-break-word">
                  {ref.definition}
                </p>
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}

export default StationWeatherMetricTerminology;
