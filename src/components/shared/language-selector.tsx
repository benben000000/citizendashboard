"use client";

import { useTransition } from "react";
import { useTranslations } from "next-intl";
import { LOCALES, type Locale } from "@/lib/i18n/translations";
import { useLocaleSwitcher } from "@/contexts/locale-context";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { Languages } from "lucide-react";

const LOCALE_SHORT_LABELS: Record<Locale, string> = {
  fil: "FILIPINO",
  en: "ENGLISH",
};

const LanguageSelector = () => {
  const t = useTranslations();
  const { locale, setLocale } = useLocaleSwitcher();
  const [, startTransition] = useTransition();

  const handleValueChange = (value: string) => {
    startTransition(() => {
      setLocale(value as Locale);
    });
  };

  return (
    <Select value={locale} onValueChange={handleValueChange}>
      <SelectTrigger
        aria-label={t("language.label")}
        className="h-9 w-11 gap-0 rounded-lg border-slate-950/10 bg-white/65 px-2 shadow-sm backdrop-blur-md sm:h-10 sm:w-[124px] sm:gap-2 sm:px-3"
      >
        <Languages className="h-4 w-4 text-slate-700 shrink-0" />

        <span className="hidden text-xs font-semibold uppercase tracking-wide text-slate-800 sm:inline w-[68px] text-left">
          {LOCALE_SHORT_LABELS[locale]}
        </span>
      </SelectTrigger>

      <SelectContent align="end" sideOffset={8}>
        {LOCALES.map((availableLocale) => (
          <SelectItem key={availableLocale} value={availableLocale}>
            <span className="w-20 text-left text-xs font-bold uppercase opacity-80 data-[state=checked]:opacity-100">
              {LOCALE_SHORT_LABELS[availableLocale]}
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default LanguageSelector;
