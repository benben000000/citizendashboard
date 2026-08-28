"use client";

import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import { useLocaleSwitcher } from "@/contexts/locale-context";
import { translations } from "@/lib/i18n/translations";

const TerminologyHeader = () => {
  const { locale } = useLocaleSwitcher();
  const copy = translations[locale].terminology;

  return (
    <>
      <div className="w-full grid grid-cols-7 items-center mb-2">
        <div className="flex items-center">
          <Link
            href="/weather"
            className="z-100 inline-flex items-center text-2xl text-muted-foreground hover:text-light"
            aria-label={copy.backToDashboard}
          >
            <ChevronLeft
              className="inline-block md:w-12 md:h-12 w-8 h-8"
              color="var(--light)"
              strokeWidth={2}
            />
          </Link>
        </div>
        <p className="font-semibol text-light text-lg md:text-4xl justify-self-center text-center col-start-2 col-span-5">
          {copy.title}
        </p>
      </div>
      <p className="text-sm text-light">{copy.subtitle}</p>
    </>
  );
};

export default TerminologyHeader;
