import { getRequestConfig } from "next-intl/server";
import { DEFAULT_TIME_ZONE } from "@/lib/i18n/i18n.config";
import { DEFAULT_LOCALE, translations } from "@/lib/i18n/translations";

export default getRequestConfig(async () => {
  const locale = DEFAULT_LOCALE;

  return {
    locale,
    timeZone: DEFAULT_TIME_ZONE,
    messages: translations[locale],
  };
});
