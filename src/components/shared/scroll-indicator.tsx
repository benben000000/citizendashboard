import { ChevronDown } from "lucide-react";

interface Props {
  show: boolean;
  onClick: () => void;
}

const ScrollIndicator = ({ show, onClick }: Props) => {
  if (!show) return null;

  return (
    <button 
      type="button" 
      onClick={onClick}
      className="absolute inset-x-0 bottom-6 flex flex-col items-center gap-2 group cursor-pointer"
    >
      {/* <span 
        className="text-sm font-medium text-light/70 drop-shadow-[0.0625rem_0.0625rem_0.25rem_rgba(0,0,0,0.3)] group-hover:text-light transition-colors"
      >
        Scroll for more
      </span> */}
      <div className="animate-bounce">
        <ChevronDown
          className="w-8 h-8 text-light/70 group-hover:text-light transition-colors"
          strokeWidth={2}
        />
      </div>
    </button>
  );
}

export default ScrollIndicator;