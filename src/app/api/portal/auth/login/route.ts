import { NextResponse } from "next/server";
import { checkCredentials, createSessionToken, PORTAL_AUTH_COOKIE } from "@/lib/auth/portal-auth";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { username, password } = body;

    if (!username || !password) {
      return NextResponse.json(
        { success: false, message: "Username and password are required" },
        { status: 400 }
      );
    }

    const isValid = checkCredentials(username, password);

    if (!isValid) {
      return NextResponse.json(
        { success: false, message: "Invalid username or password" },
        { status: 401 }
      );
    }

    // createSessionToken returns null when the portal secrets are not configured,
    // in which case authentication must not be reported as successful.
    const token = createSessionToken(username);
    if (!token) {
      return NextResponse.json(
        {
          success: false,
          message:
            "Portal authentication is not configured on this deployment. Set PORTAL_ADMIN_USER, PORTAL_ADMIN_PASSWORD and PORTAL_SECRET_SALT.",
        },
        { status: 503 }
      );
    }

    const response = NextResponse.json({
      success: true,
      message: "Authentication successful",
      user: { username, role: "Administrator" },
    });

    response.cookies.set({
      name: PORTAL_AUTH_COOKIE,
      value: token,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 24 * 60 * 60, // 24 hours
      path: "/",
    });

    return response;
  } catch (error) {
    console.error("Portal login error:", error);
    return NextResponse.json(
      { success: false, message: "Internal server error" },
      { status: 500 }
    );
  }
}
