import { useState, useEffect, useCallback, useRef } from "react";

/**
 * Coordinates the legacy two-section scroll-snap experience.
 * It initializes the visitor's position, tracks the visible section, and
 * exposes the delayed indicator and programmatic scroll action used by the UI.
 */

interface UseStickyScrollOptions {
  /** Whether the current section should be shown initially (null = not yet determined) */
  showCurrentSection: boolean | null;
  /** Callback when user scrolls from current section to info section */
  onScrollToInfo?: () => void;
}

interface UseStickyScrollReturn {
  /** Whether the scroll indicator should be shown */
  showIndicator: boolean;
  /** Scroll down to the info section */
  scrollDown: () => void;
  /** Whether the current section is visible */
  isCurrentSectionVisible: boolean;
  /** Whether the initial scroll position has been set */
  isInitialized: boolean;
}

/** Controls vertical scroll-snap state for the current and information sections. */
export function useStickyScroll({
  showCurrentSection,
  onScrollToInfo,
}: UseStickyScrollOptions): UseStickyScrollReturn {
  const [isCurrentSectionVisible, setIsCurrentSectionVisible] = useState(true);
  const [showIndicator, setShowIndicator] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const hasInitializedRef = useRef(false);
  const hasCalledScrollToInfoRef = useRef(false);

  // Initialize scroll position based on whether we should show current section
  useEffect(() => {
    if (showCurrentSection === null) return;
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    if (showCurrentSection) {
      // First-time visitor: start at top (current weather section)
      setIsCurrentSectionVisible(true);
      setIsInitialized(true);
    } else {
      // Return visitor: scroll to info section
      setIsCurrentSectionVisible(false);
      setIsInitialized(true);
    }
  }, [showCurrentSection]);

  // Handle initial scroll for return visitors after DOM is ready
  useEffect(() => {
    if (!isInitialized || showCurrentSection === null || showCurrentSection) return;
    
    const frameId = requestAnimationFrame(() => {
      const container = document.getElementById("scroll-container");
      const infoSection = document.getElementById("station-info");
      
      if (container && infoSection) {
        // Scroll the snap container to the info section
        container.scrollTo({ top: infoSection.offsetTop, behavior: "instant" });
      }
    });

    return () => cancelAnimationFrame(frameId);
  }, [isInitialized, showCurrentSection]);

  // Track scroll position to determine which section is visible
  useEffect(() => {
    const container = document.getElementById("scroll-container");
    if (!container) return;

    const handleScroll = () => {
      const scrollTop = container.scrollTop;
      const vh = window.innerHeight;
      
      // Consider "current section visible" if we're in the first half of viewport
      const isAtTop = scrollTop < vh * 0.5;
      setIsCurrentSectionVisible(isAtTop);
      
      // Call onScrollToInfo when user scrolls past the first section
      if (!isAtTop && !hasCalledScrollToInfoRef.current) {
        hasCalledScrollToInfoRef.current = true;
        onScrollToInfo?.();
      }
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, [onScrollToInfo]);

  // Show scroll indicator after delay when at current section
  useEffect(() => {
    if (!isCurrentSectionVisible) {
      setShowIndicator(false);
      return;
    }

    const timeout = setTimeout(() => setShowIndicator(true), 400);
    return () => clearTimeout(timeout);
  }, [isCurrentSectionVisible]);

  // Programmatic scroll to info section
  const scrollDown = useCallback(() => {
    const container = document.getElementById("scroll-container");
    const infoSection = document.getElementById("station-info");
    
    if (container && infoSection) {
      container.scrollTo({ 
        top: infoSection.offsetTop, 
        behavior: "smooth" 
      });
      setIsCurrentSectionVisible(false);
      onScrollToInfo?.();
    }
  }, [onScrollToInfo]);

  return { showIndicator, scrollDown, isCurrentSectionVisible, isInitialized };
}
