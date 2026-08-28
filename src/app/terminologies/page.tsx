import Disclaimer from "@/components/features/terminology/disclaimer";
import TerminologyHeader from "@/components/features/terminology/terminology-header";
import TerminologySection from "@/components/features/terminology/terminology-section";

export default function Home() {
  return (
    <div className="flex max-w-7xl mx-auto my-4 md:my-14 flex-col items-center gap-3 bg-blue min-h-screen p-8">
      <TerminologyHeader />
      <TerminologySection />
      <Disclaimer />
    </div>
  );
}
