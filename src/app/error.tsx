'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to console or error reporting service
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="flex min-h-screen min-h-[100svh] items-center justify-center">
      <div className="mx-auto max-w-md text-center px-6">
        <div className="mb-8">
          <div className="mx-auto mb-4 h-24 w-24 rounded-full bg-red-500/10 flex items-center justify-center">
            <svg
              className="h-12 w-12 text-red-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-light mb-2">
            Something went wrong!
          </h2>
          <p className="text-light/60 mb-6">
            {error.message || 'An unexpected error occurred. Please try again.'}
          </p>
        </div>

        <div className="flex flex-col gap-3">
          <Button
            onClick={reset}
            className="w-full"
            size="lg"
          >
            Try again
          </Button>
          <Button
            variant="outline"
            onClick={() => (window.location.href = '/')}
            className="w-full"
            size="lg"
          >
            Go to homepage
          </Button>
        </div>

        {error.digest && (
          <p className="mt-6 text-xs text-light/40">
            Error ID: {error.digest}
          </p>
        )}
      </div>
    </div>
  );
}
