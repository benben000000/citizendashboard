import Link from 'next/link';

export const dynamic = "force-dynamic";

export default function NotFound() {
  return (
    <div className="flex min-h-svh items-center justify-center p-6 text-white text-center">
      <div className="max-w-md mx-auto">
        <h1 className="text-8xl font-black mb-4">404</h1>
        <h2 className="text-2xl font-bold mb-2">Page Not Found</h2>
        <p className="text-white/70 mb-6 text-sm">
          Sorry, the page you are looking for does not exist or has been moved.
        </p>
        <div className="flex flex-col gap-3">
          <Link
            href="/weather"
            className="w-full py-3 px-6 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold transition-all shadow-lg text-center text-sm"
          >
            Go to Weather Dashboard
          </Link>
          <Link
            href="/prediction"
            className="w-full py-3 px-6 rounded-xl bg-white/10 hover:bg-white/20 text-white font-semibold border border-white/20 transition-all text-center text-sm"
          >
            Go to PINN-LNN Nowcast
          </Link>
        </div>
      </div>
    </div>
  );
}
