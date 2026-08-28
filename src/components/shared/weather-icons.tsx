import { WeatherCondition } from "@/lib/utils/weather";

interface WeatherIconProps {
  condition: WeatherCondition;
  className?: string;
}

const SunnyIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="sunGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#FFD93D" />
        <stop offset="100%" stopColor="#FF9500" />
      </linearGradient>
      <filter id="sunGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <mask id="sunnyMoonMask">
        <rect width="64" height="64" fill="white" />
        <circle cx="41" cy="24" r="19" fill="black" />
      </mask>
    </defs>
    <circle className="weather-sun" cx="32" cy="32" r="28" fill="url(#sunGradient)" filter="url(#sunGlow)" />
    <circle className="weather-moon" cx="32" cy="32" r="22" fill="#FFF4C2" mask="url(#sunnyMoonMask)" />
  </svg>
);

const HotIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="hotSunGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#FF6B35" />
        <stop offset="100%" stopColor="#E8300C" />
      </linearGradient>
      <filter id="hotGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <mask id="hotMoonMask">
        <rect width="64" height="64" fill="white" />
        <circle cx="41" cy="24" r="19" fill="black" />
      </mask>
    </defs>
    <circle className="weather-sun" cx="32" cy="32" r="28" fill="url(#hotSunGradient)" filter="url(#hotGlow)" />
    <circle className="weather-moon" cx="32" cy="32" r="22" fill="#FFF4C2" mask="url(#hotMoonMask)" />
  </svg>
);

const ColdIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="coldSunGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#B3E5FC" />
        <stop offset="100%" stopColor="#4FC3F7" />
      </linearGradient>
      <filter id="coldGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <mask id="coldMoonMask">
        <rect width="64" height="64" fill="white" />
        <circle cx="41" cy="24" r="19" fill="black" />
      </mask>
    </defs>
    {/* Cool-toned sun */}
    <circle className="weather-sun" cx="32" cy="32" r="28" fill="url(#coldSunGradient)" filter="url(#coldGlow)" />
    <circle className="weather-moon" cx="32" cy="32" r="22" fill="#FFF4C2" mask="url(#coldMoonMask)" />
  </svg>
);

const PartlyCloudyIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="pcSunGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#FFD93D" />
        <stop offset="100%" stopColor="#FF9500" />
      </linearGradient>
      <linearGradient id="pcCloudGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#FFFFFF" />
        <stop offset="100%" stopColor="#E0E5EC" />
      </linearGradient>
      <filter id="cloudShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity="0.15" />
      </filter>
      <filter id="sunGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <mask id="partlyCloudyMoonMask">
        <rect width="64" height="64" fill="white" />
        <circle cx="41" cy="24" r="19" fill="black" />
      </mask>
    </defs>
    {/* Sun */}
    <circle className="weather-sun" cx="32" cy="32" r="24" fill="url(#pcSunGradient)" filter="url(#sunGlow)" />
    <circle className="weather-moon" cx="32" cy="32" r="22" fill="#FFF4C2" mask="url(#partlyCloudyMoonMask)" />
    {/* Cloud scaled and positioned */}
    <path 
      d="M44.4 64h-38.8C2.7 64 0 61.3 0 57.9c0-2.8 1.9-5.1 4.5-5.7-.1-.5-.2-1-.2-1.4 0-4.7 3.7-8.5 8.2-8.5.9 0 1.7.2 2.5.4 2.1-3.9 6.2-6.6 10.9-6.6 7.4 0 13.4 6 13.4 13.5 0 .2 0 .6-.1.8 4 .6 7.2 4.2 7.2 8.3 0 3.5-2 6.3-4.4 6.3z" 
      fill="url(#pcCloudGradient)" 
      filter="url(#cloudShadow)"
    />
  </svg>
);

const CloudyIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="cloudySunGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#FFD93D" />
        <stop offset="100%" stopColor="#FF9500" />
      </linearGradient>
      <linearGradient id="cloudyCloudGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#FFFFFF" />
        <stop offset="100%" stopColor="#E0E5EC" />
      </linearGradient>
      <filter id="cloudyShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity="0.15" />
      </filter>
      <filter id="cloudySunGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <mask id="cloudyMoonMask">
        <rect width="64" height="64" fill="white" />
        <circle cx="41" cy="24" r="19" fill="black" />
      </mask>
    </defs>
    {/* Sun */}
    <circle className="weather-sun" cx="32" cy="32" r="20" fill="url(#cloudySunGradient)" filter="url(#cloudySunGlow)" />
    <circle className="weather-moon" cx="32" cy="32" r="22" fill="#FFF4C2" mask="url(#cloudyMoonMask)" />
    {/* Cloud at back right - positioned at middle right, offset downward */}
    <g transform="translate(28, 4) scale(0.55)">
      <path 
        d="M56.5 40h-49C3.4 40 0 36.6 0 32.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 18.1 10.1 13.5 16 13.5c1.1 0 2.2.2 3.2.5C21.8 8.5 27.4 5 34 5c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z" 
        fill="url(#cloudyCloudGradient)" 
        filter="url(#cloudyShadow)"
        opacity="0.9"
      />
    </g>
    {/* Cloud at front bottom - same size as back cloud */}
    <g transform="translate(0, 18) scale(0.55)">
      <path 
        d="M56.5 64h-49C3.4 64 0 60.6 0 56.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 42.1 10.1 37.5 16 37.5c1.1 0 2.2.2 3.2.5C21.8 32.5 27.4 29 34 29c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z" 
        fill="url(#cloudyCloudGradient)" 
        filter="url(#cloudyShadow)"
      />
    </g>
  </svg>
);

const RainIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="rCB" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#B8CEDA" /><stop offset="100%" stopColor="#9AB5C5" />
      </linearGradient>
      <linearGradient id="rCM" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#D2E0EA" /><stop offset="100%" stopColor="#BACCD8" />
      </linearGradient>
      <linearGradient id="rCF" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#EDF3F7" /><stop offset="100%" stopColor="#D0DDE7" />
      </linearGradient>
    </defs>
    {/* Back cloud — smaller, peeking right */}
    <g transform="translate(16, 0) scale(0.78)" opacity="0.7">
      <path d="M56.5 40h-49C3.4 40 0 36.6 0 32.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 18.1 10.1 13.5 16 13.5c1.1 0 2.2.2 3.2.5C21.8 8.5 27.4 5 34 5c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z"
        fill="url(#rCB)" />
    </g>
    <g transform="translate(0, 4) scale(0.88)" opacity="0.85">
      <path d="M56.5 40h-49C3.4 40 0 36.6 0 32.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 18.1 10.1 13.5 16 13.5c1.1 0 2.2.2 3.2.5C21.8 8.5 27.4 5 34 5c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z"
        fill="url(#rCM)" />
    </g>
    {/* Front cloud — full size, centered */}
    <g transform="translate(3.75, 6)">
      <path d="M56.5 40h-49C3.4 40 0 36.6 0 32.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 18.1 10.1 13.5 16 13.5c1.1 0 2.2.2 3.2.5C21.8 8.5 27.4 5 34 5c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z"
        fill="url(#rCF)" />
    </g>
    {/* Rain streaks */}
    <g stroke="#247DB8" strokeWidth="2.4" strokeLinecap="round" opacity="0.95">
      <line x1="14" y1="49" x2="11" y2="60" />
      <line x1="23" y1="47" x2="20" y2="58" />
      <line x1="32" y1="49" x2="29" y2="60" />
      <line x1="41" y1="47" x2="38" y2="58" />
      <line x1="50" y1="49" x2="47" y2="60" />
    </g>
  </svg>
);

const StormIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="sCB" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#374355" /><stop offset="100%" stopColor="#222D3D" />
      </linearGradient>
      <linearGradient id="sCM" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#4A5870" /><stop offset="100%" stopColor="#333F54" />
      </linearGradient>
      <linearGradient id="sCF" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stopColor="#5E6E88" /><stop offset="100%" stopColor="#44536A" />
      </linearGradient>
      <linearGradient id="boltG" x1="0%" y1="0%" x2="20%" y2="100%">
        <stop offset="0%" stopColor="#FFF176" /><stop offset="100%" stopColor="#FFB300" />
      </linearGradient>
    </defs>
    {/* Back cloud */}
    <g transform="translate(16, 0) scale(0.78)" opacity="0.6">
      <path d="M56.5 40h-49C3.4 40 0 36.6 0 32.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 18.1 10.1 13.5 16 13.5c1.1 0 2.2.2 3.2.5C21.8 8.5 27.4 5 34 5c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z"
        fill="url(#sCB)" />
    </g>
    {/* Mid cloud */}
    <g transform="translate(0, 4) scale(0.88)" opacity="0.8">
      <path d="M56.5 40h-49C3.4 40 0 36.6 0 32.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 18.1 10.1 13.5 16 13.5c1.1 0 2.2.2 3.2.5C21.8 8.5 27.4 5 34 5c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z"
        fill="url(#sCM)" />
    </g>
    {/* Front cloud */}
    <g transform="translate(3.75, 6)">
      <path d="M56.5 40h-49C3.4 40 0 36.6 0 32.5c0-3.5 2.4-6.4 5.7-7.2-.1-.6-.2-1.2-.2-1.8C5.5 18.1 10.1 13.5 16 13.5c1.1 0 2.2.2 3.2.5C21.8 8.5 27.4 5 34 5c9.4 0 17 7.6 17 17 0 .3 0 .7-.1 1 5.1.8 9.1 5.3 9.1 10.5 0 4-1.5 6.5-3.5 6.5z"
        fill="url(#sCF)" />
    </g>
    {/* Lightning bolt */}
    <path d="M34 40 L27 52 H33 L25 63 L43 47 H36 Z" fill="url(#boltG)" />
    {/* Rain streaks flanking bolt */}
    <g stroke="#247DB8" strokeWidth="2.4" strokeLinecap="round" opacity="0.9">
      <line x1="13" y1="48" x2="10" y2="58" />
      <line x1="21" y1="46" x2="18" y2="56" />
      <line x1="45" y1="46" x2="42" y2="56" />
      <line x1="53" y1="48" x2="50" y2="58" />
    </g>
  </svg>
);

const HighUVIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 64 64" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="uvSunGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#FF6B6B" />
        <stop offset="50%" stopColor="#FF8E53" />
        <stop offset="100%" stopColor="#FFD93D" />
      </linearGradient>
      <filter id="uvGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <mask id="highUVMoonMask">
        <rect width="64" height="64" fill="white" />
        <circle cx="41" cy="24" r="19" fill="black" />
      </mask>
    </defs>
    <circle className="weather-sun" cx="32" cy="32" r="28" fill="url(#uvSunGradient)" filter="url(#uvGlow)" />
    <circle className="weather-moon" cx="32" cy="32" r="22" fill="#FFF4C2" mask="url(#highUVMoonMask)" />
  </svg>
);

const WeatherIcon = ({ condition, className }: WeatherIconProps) => {
  switch (condition) {
    case "sunny":
      return <SunnyIcon className={className} />;
    case "hot":
      return <CloudyIcon className={className} />;
    case "cold":
      return <ColdIcon className={className} />;
    case "partly-cloudy":
      return <PartlyCloudyIcon className={className} />;
    case "cloudy":
      return <CloudyIcon className={className} />;
    case "rain":
      return <RainIcon className={className} />;
    case "storm":
      return <StormIcon className={className} />;
    case "high-uv":
      return <CloudyIcon className={className} />;
    default:
      return <SunnyIcon className={className} />;
  }

};

export default WeatherIcon;
