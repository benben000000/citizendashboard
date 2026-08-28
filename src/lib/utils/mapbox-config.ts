/**
 * Mapbox GL JS configuration utilities
 * 
 * This module provides utilities to:
 * - Disable Mapbox telemetry to prevent CORS errors
 * - Suppress WebGL deprecation warnings
 */

/**
 * Disables Mapbox telemetry by intercepting requests to the events endpoint.
 * This prevents CORS errors when ad blockers or privacy tools block the telemetry endpoint.
 */
export function disableMapboxTelemetry(): () => void {
  if (typeof window === 'undefined') {
    return () => {}; // No-op on server
  }

  // Store original fetch
  const originalFetch = window.fetch;
  const originalXHROpen = XMLHttpRequest.prototype.open;
  const originalXHRSend = XMLHttpRequest.prototype.send;

  // Block fetch requests to Mapbox events endpoint
  window.fetch = function (input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    
    if (url.includes('events.mapbox.com')) {
      // Return a rejected promise to prevent the request
      return Promise.reject(new Error('Mapbox telemetry disabled'));
    }
    
    return originalFetch.call(this, input, init);
  };

  // Block XMLHttpRequest to Mapbox events endpoint
  let xhrUrl = '';
  XMLHttpRequest.prototype.open = function (method: string, url: string | URL, ...args: any[]) {
    xhrUrl = typeof url === 'string' ? url : url.href;
    return (originalXHROpen as any).call(this, method, url, ...(args as any[]));
  };

  XMLHttpRequest.prototype.send = function (...args: any[]) {
    if (xhrUrl.includes('events.mapbox.com')) {
      // Abort the request silently
      return;
    }
    return originalXHRSend.call(this, ...args);
  };

  // Return cleanup function
  return () => {
    window.fetch = originalFetch;
    XMLHttpRequest.prototype.open = originalXHROpen;
    XMLHttpRequest.prototype.send = originalXHRSend;
  };
}

/**
 * Suppresses WebGL deprecation warnings in the console.
 * These warnings come from Mapbox GL JS internals and are mostly harmless.
 */
export function suppressWebGLWarnings(): () => void {
  if (typeof window === 'undefined') {
    return () => {}; // No-op on server
  }

  const originalWarn = console.warn;
  const originalError = console.error;

  // Filter out WebGL deprecation warnings and Mapbox style warnings
  const webglWarningPatterns = [
    /WEBGL_debug_renderer_info is deprecated/,
    /texSubImage: Alpha-premult and y-flip are deprecated for non-DOM-Element uploads/,
    /Cross-Origin Request Blocked.*events\.mapbox\.com/,
    /featureNamespace.*selector is not associated to the same source/,
    /Cutoff is currently disabled on terrain/,
  ];

  console.warn = function (...args: any[]) {
    const message = args.join(' ');
    if (!webglWarningPatterns.some(pattern => pattern.test(message))) {
      originalWarn.apply(console, args);
    }
  };

  console.error = function (...args: any[]) {
    const message = args.join(' ');
    if (!webglWarningPatterns.some(pattern => pattern.test(message))) {
      originalError.apply(console, args);
    }
  };

  // Return cleanup function
  return () => {
    console.warn = originalWarn;
    console.error = originalError;
  };
}

/**
 * Initializes all Mapbox-related fixes.
 * Call this once when your app loads or when the map component mounts.
 * 
 * @returns Cleanup function to restore original behavior
 */
export function initializeMapboxFixes(): () => void {
  const cleanupTelemetry = disableMapboxTelemetry();
  const cleanupWarnings = suppressWebGLWarnings();

  return () => {
    cleanupTelemetry();
    cleanupWarnings();
  };
}
