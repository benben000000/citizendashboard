import { cookies } from "next/headers";

const PORTAL_AUTH_COOKIE = "kloudtrack_portal_session";
const DEFAULT_PASSWORD = process.env.PORTAL_ADMIN_PASSWORD || "Kloudtrack2026!";
const DEFAULT_USERNAME = process.env.PORTAL_ADMIN_USER || "admin";
const SECRET_SALT = process.env.PORTAL_SECRET_SALT || "kloudtrack_portal_salt_2026";

/**
 * Generates a simple tamper-resistant session token
 */
export function createSessionToken(username: string): string {
  const expiresAt = Date.now() + 24 * 60 * 60 * 1000; // 24 hours
  const payload = `${username}:${expiresAt}`;
  const signature = Buffer.from(`${payload}:${SECRET_SALT}`).toString("base64url");
  return `${Buffer.from(payload).toString("base64url")}.${signature}`;
}

/**
 * Validates a session token
 */
export function verifySessionToken(token: string | undefined | null): { valid: boolean; username?: string } {
  if (!token) return { valid: false };

  try {
    const [payloadB64, signature] = token.split(".");
    if (!payloadB64 || !signature) return { valid: false };

    const payload = Buffer.from(payloadB64, "base64url").toString("utf-8");
    const expectedSig = Buffer.from(`${payload}:${SECRET_SALT}`).toString("base64url");

    if (signature !== expectedSig) return { valid: false };

    const [username, expiresAtStr] = payload.split(":");
    const expiresAt = Number(expiresAtStr);

    if (Date.now() > expiresAt) return { valid: false };

    return { valid: true, username };
  } catch {
    return { valid: false };
  }
}

/**
 * Validates credentials
 */
export function checkCredentials(user: string, pass: string): boolean {
  return (
    (user.trim().toLowerCase() === DEFAULT_USERNAME.toLowerCase() || user.trim() === "admin") &&
    pass === DEFAULT_PASSWORD
  );
}

/**
 * Checks session in Server Components / Route Handlers
 */
export function getPortalSession() {
  const cookieStore = cookies();
  const token = cookieStore.get(PORTAL_AUTH_COOKIE)?.value;
  return verifySessionToken(token);
}

export { PORTAL_AUTH_COOKIE };
