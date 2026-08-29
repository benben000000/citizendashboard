import { NextResponse } from "next/server";
import { getPortalSession } from "@/lib/auth/portal-auth";

export const dynamic = "force-dynamic";

export async function GET() {
  const session = getPortalSession();

  if (!session.valid) {
    return NextResponse.json(
      { authenticated: false },
      { status: 401 }
    );
  }

  return NextResponse.json({
    authenticated: true,
    user: {
      username: session.username || "admin",
      role: "Administrator",
    },
  });
}
