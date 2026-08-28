/**
 * i18n Configuration
 * 
 * Change DEFAULT_LANGUAGE here to switch the default language for the entire application.
 * Supported languages: 'en' (English), 'fil' (Filipino)
 */

export const DEFAULT_LANGUAGE = "fil" as const;
export const DEFAULT_TIME_ZONE = "Asia/Manila";

// Supported languages in the application
export const SUPPORTED_LANGUAGES = ["en", "fil"] as const;

// Language display labels
export const LANGUAGE_LABELS: Record<typeof SUPPORTED_LANGUAGES[number], string> = {
  en: "English",
  fil: "Filipino",
};
