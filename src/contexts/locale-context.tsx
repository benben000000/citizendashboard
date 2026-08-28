"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  startTransition,
  type ReactNode,
} from "react";
import { NextIntlClientProvider } from "next-intl";
import { DEFAULT_TIME_ZONE } from "@/lib/i18n/i18n.config";
import {
  DEFAULT_LOCALE,
  LOCALES,
  translations,
  type Locale,
} from "@/lib/i18n/translations";

const LOCALE_STORAGE_KEY = "kloudtrack_locale";

interface LocaleContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
}

const LocaleContext = createContext<LocaleContextValue | null>(null);

const isLocale = (value: string | null): value is Locale => {
  return !!value && LOCALES.includes(value as Locale);
};

export const LocaleProvider = ({ children }: { children: ReactNode }) => {
  const [locale, setLocaleState] = useState<Locale>(DEFAULT_LOCALE);

  useEffect(() => {
    const storedLocale = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    if (isLocale(storedLocale)) {
      setLocaleState(storedLocale);
      document.documentElement.lang = storedLocale;
    }
  }, []);

  const setLocale = (nextLocale: Locale) => {
    startTransition(() => {
      setLocaleState(nextLocale);
    });
    window.localStorage.setItem(LOCALE_STORAGE_KEY, nextLocale);
    document.documentElement.lang = nextLocale;
  };

  const value = useMemo(
    () => ({
      locale,
      setLocale,
    }),
    [locale],
  );

  return (
    <LocaleContext.Provider value={value}>
      <NextIntlClientProvider
        locale={locale}
        timeZone={DEFAULT_TIME_ZONE}
        messages={translations[locale]}
      >
        {children}
      </NextIntlClientProvider>
    </LocaleContext.Provider>
  );
};

export const useLocaleSwitcher = () => {
  const context = useContext(LocaleContext);
  if (!context) {
    throw new Error("useLocaleSwitcher must be used within LocaleProvider");
  }
  return context;
};
