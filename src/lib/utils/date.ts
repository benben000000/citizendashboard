import { DEFAULT_LOCALE, type Locale } from "@/lib/i18n/translations"

type FormatResult = {
  formatted: string
  relative: string
}

const INTL_DATE_LOCALES: Record<Locale, string> = {
  en: "en-US",
  fil: "fil-PH",
}

export const getIntlDateLocale = (locale: Locale): string =>
  INTL_DATE_LOCALES[locale] ?? INTL_DATE_LOCALES[DEFAULT_LOCALE]

const DATE_COPY: Record<
  Locale,
  {
    noRecentData: string
    unknown: string
    invalidDate: string
    justNow: string
    minutesAgo: (value: number) => string
    hoursAgo: (value: number) => string
    daysAgo: (value: number) => string
  }
> = {
  en: {
    noRecentData: "No recent data",
    unknown: "Unknown",
    invalidDate: "Invalid date",
    justNow: "Just now",
    minutesAgo: (value) => `${value}m ago`,
    hoursAgo: (value) => `${value}h ago`,
    daysAgo: (value) => `${value}d ago`,
  },
  fil: {
    noRecentData: "Walang kamakailang datos",
    unknown: "Hindi alam",
    invalidDate: "Di-wastong petsa",
    justNow: "Ngayon lang",
    minutesAgo: (value) => `${value} min ang nakalipas`,
    hoursAgo: (value) => `${value} oras ang nakalipas`,
    daysAgo: (value) => `${value} araw ang nakalipas`,
  },
}

const resolveDateOptions = (
  options: Intl.DateTimeFormatOptions | undefined,
  locale: Locale,
): Intl.DateTimeFormatOptions => {
  const resolvedOptions =
    options ?? {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    }

  if (locale === "fil") {
    return {
      ...resolvedOptions,
      weekday: resolvedOptions.weekday === "short" ? "long" : resolvedOptions.weekday,
      month: resolvedOptions.month === "short" ? "long" : resolvedOptions.month,
    }
  }

  return resolvedOptions
}

// Overload signatures
export function formatDate(dateString?: string, locale?: Locale): FormatResult
export function formatDate(
  dateString: string | undefined,
  options: Intl.DateTimeFormatOptions,
  locale?: Locale
): FormatResult

// Implementation
export function formatDate(
  dateString?: string,
  optionsOrLocale?: Intl.DateTimeFormatOptions | Locale,
  locale: Locale = DEFAULT_LOCALE
): FormatResult {
  const options = typeof optionsOrLocale === "string" ? undefined : optionsOrLocale
  const resolvedLocale = typeof optionsOrLocale === "string" ? optionsOrLocale : locale
  const copy = DATE_COPY[resolvedLocale] ?? DATE_COPY[DEFAULT_LOCALE]

  if (!dateString) {
    return { formatted: copy.noRecentData, relative: copy.unknown }
  }

  try {
    const date = new Date(dateString)
    const now = new Date()

    const diffMs = now.getTime() - date.getTime()
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffMins = Math.floor(diffMs / (1000 * 60))

    let relative = ""
    if (diffMins < 1) relative = copy.justNow
    else if (diffMins < 60) relative = copy.minutesAgo(diffMins)
    else if (diffHours < 24) relative = copy.hoursAgo(diffHours)
    else relative = copy.daysAgo(Math.floor(diffHours / 24))

    const formatted = date.toLocaleString(
      getIntlDateLocale(resolvedLocale),
      resolveDateOptions(options, resolvedLocale)
    )

    return { formatted, relative }
  } catch {
    return { formatted: copy.invalidDate, relative: copy.unknown }
  }
}
