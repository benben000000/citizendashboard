/**
 * Simple in-memory cache with TTL (Time To Live)
 */
export class InMemoryCache<T> {
  private cache = new Map<string, { data: T; expiresAt: number }>();
  private defaultTTL: number; // in seconds

  constructor(defaultTTL: number = 60) {
    this.defaultTTL = defaultTTL;
  }

  /**
   * Set a value in the cache
   * @param key Cache key
   * @param value Value to cache
   * @param ttl Time to live in seconds (optional, uses default if not provided)
   */
  set(key: string, value: T, ttl?: number): void {
    const ttlToUse = ttl ?? this.defaultTTL;
    const expiresAt = Date.now() + ttlToUse * 1000;
    this.cache.set(key, { data: value, expiresAt });
  }

  /**
   * Get a value from the cache
   * @param key Cache key
   * @returns Cached value or undefined if not found or expired
   */
  get(key: string): T | undefined {
    const cached = this.cache.get(key);

    if (!cached) {
      return undefined;
    }

    // Check if expired
    if (Date.now() > cached.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    return cached.data;
  }

  /**
   * Check if a key exists and is not expired
   */
  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  /**
   * Delete a specific key from cache
   */
  delete(key: string): void {
    this.cache.delete(key);
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get the number of cached items (including expired ones)
   */
  size(): number {
    return this.cache.size;
  }

  /**
   * Clean up expired entries
   */
  cleanup(): void {
    const now = Date.now();
    const entries = Array.from(this.cache.entries());
    for (const [key, value] of entries) {
      if (now > value.expiresAt) {
        this.cache.delete(key);
      }
    }
  }
}
