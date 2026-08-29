import { redirect } from "next/navigation";
import { getPortalSession } from "@/lib/auth/portal-auth";
import { PortalClientDashboard } from "@/components/features/portal/portal-client-dashboard";

export const dynamic = "force-dynamic";

export default async function PortalPage() {
  const session = getPortalSession();

  if (!session.valid) {
    redirect("/portal/login");
  }

  return <PortalClientDashboard username={session.username || "admin"} />;
}
