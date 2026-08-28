import { en } from "@/lib/i18n/messages/en";
import { fil } from "@/lib/i18n/messages/fil";
import { DEFAULT_LANGUAGE, LANGUAGE_LABELS } from "@/lib/i18n/i18n.config";

export const DEFAULT_LOCALE = DEFAULT_LANGUAGE;

export const translations = {
  fil,
  en,
} as const;

export type Locale = keyof typeof translations;
export const LOCALES = Object.keys(translations) as Locale[];
export const LOCALE_LABELS = LANGUAGE_LABELS;
type DeepStringValues<TValue> = TValue extends string
  ? string
  : {
      readonly [TKey in keyof TValue]: DeepStringValues<TValue[TKey]>;
    };

export type Messages = DeepStringValues<typeof en>;
export type TranslationDictionary = Messages;
export type WeatherWarningCopyKey = keyof Messages["weatherWarnings"]["warnings"];
export type WeatherActionCopyKey = keyof Messages["weatherWarnings"]["actions"];
export type WeatherWarningCommonKey = keyof Messages["weatherWarnings"]["common"];

type DotPrefix<TPrefix extends string, TKey extends string> =
  TPrefix extends "" ? TKey : `${TPrefix}.${TKey}`;

export type TranslationKey<TValue = Messages, TPrefix extends string = ""> = {
  [TKey in keyof TValue & string]: TValue[TKey] extends string
    ? DotPrefix<TPrefix, TKey>
    : TranslationKey<TValue[TKey], DotPrefix<TPrefix, TKey>>;
}[keyof TValue & string];

const getLocaleCopy = (locale: Locale = DEFAULT_LOCALE) => {
  return translations[locale] ?? translations[DEFAULT_LOCALE];
};

export const getWeatherWarningCopy = (
  key: WeatherWarningCopyKey,
  locale: Locale = DEFAULT_LOCALE,
) =>
  getLocaleCopy(locale).weatherWarnings.warnings[key] ??
  translations[DEFAULT_LOCALE].weatherWarnings.warnings[key];

export const getWeatherActionCopy = (
  key: WeatherActionCopyKey,
  locale: Locale = DEFAULT_LOCALE,
) =>
  getLocaleCopy(locale).weatherWarnings.actions[key] ??
  translations[DEFAULT_LOCALE].weatherWarnings.actions[key];

export const getWeatherCommonCopy = (
  key: WeatherWarningCommonKey,
  locale: Locale = DEFAULT_LOCALE,
) =>
  getLocaleCopy(locale).weatherWarnings.common[key] ??
  translations[DEFAULT_LOCALE].weatherWarnings.common[key];
