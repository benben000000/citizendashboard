import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const LEARN_MORE_URL = "https://homepage.kloudtechsea.com/";
const CONTACT_URL = "https://homepage.kloudtechsea.com/contact-us";

export default function CustomFooter() {
  return (
    <footer className="border-t border-border/50 bg-muted/30">
      <div className="max-w-360 mx-auto px-5 md:px-10 py-8 md:py-12">

        {/* Legal */}
        <div className="flex flex-col md:flex-row md:justify-between gap-1 text-xs text-foreground/70">
          <p>© {new Date().getFullYear()} Kloudtech Corp. All rights reserved.</p>
          <p>Made in the Philippines</p>
        </div>
      </div>
    </footer>
  );
}