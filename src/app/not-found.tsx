import Link from 'next/link';
import { Button } from '@/components/ui/button';

export default function NotFound() {
  return (
    <div className="flex min-h-svh items-center justify-center">
      <div className="mx-auto max-w-md text-center px-6">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-light mb-4">404</h1>
          <h2 className="text-3xl font-bold text-light mb-2">
            Page not found
          </h2>
          <p className="text-light/60 mb-6">
            Sorry, we couldn&apos;t find the page you&apos;re looking for.
          </p>
        </div>

        <div className="flex flex-col gap-3">
          <Link href="/weather">
            <Button
              className="w-full"
              size="lg"
            >
              Go to homepage
            </Button>
          </Link>
          <Link href="/terminologies">
            <Button
              variant="outline"
              className="w-full"
              size="lg"
            >
              View terminologies
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
