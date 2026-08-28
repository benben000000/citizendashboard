import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

export default async function Home({
  searchParams,
}: {
  searchParams: { lat?: string; lon?: string; location?: string };
}) {
  const params = new URLSearchParams();

  if (searchParams.lat) params.set("lat", searchParams.lat);
  if (searchParams.lon) params.set("lon", searchParams.lon);
  if (searchParams.location) params.set("location", searchParams.location);

  const query = params.toString();

  redirect(query ? `/weather?${query}` : "/weather");
}
