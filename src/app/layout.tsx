import type { Metadata } from "next";
import { Suspense } from "react";
import "./globals.css";
import { Inter } from "next/font/google";
import DiffusedCloudBackground from "../components/shared/diffused-cloud-background";
import { ThemeProvider } from "@/contexts/theme-context";
import { LocaleProvider } from "@/contexts/locale-context";
import { DEFAULT_LOCALE } from "@/lib/i18n/translations";
import RouteTransitionLoader from "@/components/shared/route-transition-loader";

const inter = Inter({ subsets: ["latin"] });

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://citizen.kloudtechsea.com";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: "Kloudtrack", template: "%s | Kloudtrack" },
  description:
    "Kloudtrack is a network of real-time, hyper-localized weather monitoring stations and a web app that helps local authorities and communities address weather-related challenges.",
  keywords: [
    "Kloudtrack",
    "weather",
    "hyperlocal",
    "telemetry",
    "weather station",
  ],
  authors: [{ name: "Kloudtech" }],
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  openGraph: {
    title: "Kloudtrack",
    description:
      "Real-time, hyper-localized weather monitoring and insights for communities and authorities.",
    url: SITE_URL,
    siteName: "Kloudtrack",
    images: [
      {
        url: `${SITE_URL}/images/banner.png`,
        width: 1200,
        height: 630,
        alt: "Kloudtrack — hyper-local weather monitoring",
        type: "image/png",
      },
    ],
    locale: DEFAULT_LOCALE,
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Kloudtrack",
    description:
      "Real-time, hyper-localized weather monitoring and insights for communities and authorities.",
    images: [`${SITE_URL}/images/banner.png`],
  },
  icons: {
    icon: "/icons/logo.png",
    apple: "/icons/logo.png",
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "Organization",
  name: "Kloudtrack",
  url: SITE_URL,
  logo: `${SITE_URL}/icons/logo.png`,
  sameAs: ["https://twitter.com/kloudtech"],
  description:
    "Kloudtrack is a network of real-time, hyper-localized weather monitoring stations and a web app that helps local authorities and communities address weather-related challenges.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang={DEFAULT_LOCALE}>
      <body className={`${inter.className} min-h-screen min-h-[100svh]`}>
        <ThemeProvider>
          <LocaleProvider>
            <div className="app-sky-gradient relative flex flex-col min-h-screen min-h-[100svh]">
              <DiffusedCloudBackground />
              <script
                key="ldjson"
                type="application/ld+json"
                // eslint-disable-next-line react/no-danger
                dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
              />
              <div className="relative z-10 flex flex-col">{children}</div>
              <Suspense fallback={null}>
                <RouteTransitionLoader />
              </Suspense>
            </div>
          </LocaleProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
