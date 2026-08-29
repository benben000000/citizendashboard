/**
 * High-Performance In-Memory Cache with TTL, SWR, and LRU Eviction
 */
export class InMemoryCache<T> {
  private cache = new Map<string, { data: T; expiresAt: number; staleAt: number }>();
  private defaultTTL: number; // in seconds
  private maxEntries: number;

  constructor(defaultTTL: number = 60, maxEntries: number = 500) {
    this.defaultTTL = defaultTTL;
    this.maxEntries = maxEntries;
  }

  /**
   * Set a value in the cache with automatic LRU eviction
   */
  set(key: string, value: T, ttl?: number): void {
    if (this.cache.size >= this.maxEntries && !this.cache.has(key)) {
      // Evict oldest entry (LRU via Map key order)
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey !== undefined) {
        this.cache.delete(oldestKey);
      }
    }

    const ttlToUse = ttl ?? this.defaultTTL;
    const now = Date.now();
    const staleAt = now + ttlToUse * 1000;
    const expiresAt = now + ttlToUse * 2000; // Allow stale read up to 2x TTL
    this.cache.set(key, { data: value, expiresAt, staleAt });
  }

  /**
   * Get a value from the cache with O(1) retrieval
   */
  get(key: string): T | undefined {
    const cached = this.cache.get(key);
    if (!cached) return undefined;

    const now = Date.now();
    if (now > cached.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    // Refresh LRU order on access
    this.cache.delete(key);
    this.cache.set(key, cached);

    return cached.data;
  }

  /**
   * Check if cache entry is stale (useful for background revalidation)
   */
  isStale(key: string): boolean {
    const cached = this.cache.get(key);
    if (!cached) return true;
    return Date.now() > cached.staleAt;
  }

  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  delete(key: string): void {
    this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }

  cleanup(): void {
    const now = Date.now();
    for (const [key, value] of this.cache.entries()) {
      if (now > value.expiresAt) {
        this.cache.delete(key);
      }
    }
  }
}
