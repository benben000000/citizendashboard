import createNextIntlPlugin from "next-intl/plugin";

/** @type {import('next').NextConfig} */
const nextConfig = {
  optimizeFonts: false,
  compress: true,
  poweredByHeader: false,
  experimental: {
    optimizePackageImports: [
      "lucide-react",
      "recharts",
      "framer-motion",
      "clsx",
      "tailwind-merge",
      "date-fns",
    ],
  },
  webpack(config) {
    config.infrastructureLogging = { level: "error" }; // hides cache warnings
    return config;
  },
};

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

export default withNextIntl(nextConfig);
