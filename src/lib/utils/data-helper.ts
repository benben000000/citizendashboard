/**
 * Data transformation and formatting utilities
 */

/**
 * Round a number to two decimal places
 * Returns null if input is null, undefined, or NaN
 */
export function toTwoDecimalPlaces(value: number | null | undefined): number | null {
  if (value === null || value === undefined || isNaN(value)) {
    return null;
  }
  return Math.round(value * 100) / 100;
}

/**
 * Safely extract a nested property from an object
 * Returns null if path doesn't exist
 */
export function safeGet<T>(obj: unknown, path: string): T | null {
  const keys = path.split('.');
  let current: unknown = obj;

  for (const key of keys) {
    if (current === null || current === undefined) {
      return null;
    }
    if (typeof current === 'object' && current !== null) {
      current = (current as Record<string, unknown>)[key];
    } else {
      return null;
    }
  }

  return (current as T) ?? null;
}
