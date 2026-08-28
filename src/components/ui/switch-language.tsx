"use client";

import * as React from "react";
import * as SwitchPrimitives from "@radix-ui/react-switch";
import { cn } from "@/lib/utils/cn";

const SwitchLanguage = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => {
  const [isChecked, setIsChecked] = React.useState(false);

  return (
    <SwitchPrimitives.Root
      className={cn(
        "peer inline-flex h-8 w-14 shrink-0 cursor-pointer items-center justify-start rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-[#FBD008] data-[state=unchecked]:bg-input",
        className
      )}
      {...props}
      ref={ref}
      checked={isChecked}
      onCheckedChange={setIsChecked}
    >
      <SwitchPrimitives.Thumb
        className={cn(
          "pointer-events-none flex items-center justify-center h-8 w-8 rounded-full bg-background ring-0 transition-transform data-[state=checked]:translate-x-6 data-[state=unchecked]:translate-x-0"
        )}
      >
        <span className="font-bold">{isChecked ? "F" : "E"}</span>
      </SwitchPrimitives.Thumb>
    </SwitchPrimitives.Root>
  );
});

SwitchLanguage.displayName = SwitchPrimitives.Root.displayName;

export { SwitchLanguage };
