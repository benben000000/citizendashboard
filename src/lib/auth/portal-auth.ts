import { cookies } from "next/headers";
import crypto from "crypto";

const PORTAL_AUTH_COOKIE = "kloudtrack_portal_session";
const SESSION_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

/**
 * Portal authentication.
 *
 * SECURITY CHANGES (see AUDIT.md)
 * -------------------------------
 * 1. The session "signature" was `base64url(payload + ":" + SECRET_SALT)`.
 *    That is an ENCODING, not a MAC. Because the salt is a compile-time
 *    constant, anyone who can read the source can mint a valid session for any
 *    username and any expiry. It is replaced here with a real HMAC-SHA256.
 * 2. The admin password and salt had hardcoded fallbacks
 *    (`Kloudtrack2026!` / `kloudtrack_portal_salt_2026`) committed to the
 *    repository. Configuration now FAILS CLOSED: if the secrets are unset the
 *    portal is unreachable rather than protected by a published password.
 * 3. `checkCredentials` accepted the literal username "admin" regardless of
 *    the configured user, so a configured non-admin account was still reachable
 *    as "admin". Only the exact configured username is accepted now.
 * 4. Signature and password comparison use `timingSafeEqual`.
 */

/** Returns the configured secret, or null when the portal is not configured. */
function readSecret(name: "PORTAL_ADMIN_PASSWORD" | "PORTAL_SECRET_SALT"): string | null {
  const value = process.env[name];
  if (!value || value.trim().length === 0) return null;
  return value;
}

/** True when the portal has usable credentials configured. */
export function isPortalConfigured(): boolean {
  return readSecret("PORTAL_ADMIN_PASSWORD") !== null && readSecret("PORTAL_SECRET_SALT") !== null;
}

function getAdminUser(): string | null {
  const value = process.env.PORTAL_ADMIN_USER;
  if (!value || value.trim().length === 0) return null;
  return value.trim();
}

function sign(payload: string): string {
  const salt = readSecret("PORTAL_SECRET_SALT") as string;
  return crypto.createHmac("sha256", salt).update(payload).digest("base64url");
}

function safeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a, "utf-8");
  const bufB = Buffer.from(b, "utf-8");
  if (bufA.length !== bufB.length) return false;
  return crypto.timingSafeEqual(bufA, bufB);
}

/** Generates an HMAC-signed session token. Returns null if unconfigured. */
export function createSessionToken(username: string): string | null {
  const salt = readSecret("PORTAL_SECRET_SALT");
  if (salt === null) return null;
  const expiresAt = Date.now() + SESSION_TTL_MS;
  const payload = `${username}:${expiresAt}`;
  const encoded = Buffer.from(payload, "utf-8").toString("base64url");
  return `${encoded}.${sign(payload)}`;
}

/** Validates a session token's HMAC and expiry. */
export function verifySessionToken(
  token: string | undefined | null
): { valid: boolean; username?: string } {
  if (!token) return { valid: false };
  if (readSecret("PORTAL_SECRET_SALT") === null) return { valid: false };

  try {
    const [payloadB64, signature] = token.split(".");
    if (!payloadB64 || !signature) return { valid: false };

    const payload = Buffer.from(payloadB64, "base64url").toString("utf-8");
    if (!safeEqual(signature, sign(payload))) return { valid: false };

    const [username, expiresAtStr] = payload.split(":");
    const expiresAt = Number(expiresAtStr);
    if (!Number.isFinite(expiresAt)) return { valid: false };
    if (Date.now() > expiresAt) return { valid: false };

    return { valid: true, username };
  } catch {
    return { valid: false };
  }
}

/**
 * Validates credentials against the configured admin account only.
 * Fails closed when the portal is not configured.
 */
export function checkCredentials(user: string, pass: string): boolean {
  const adminUser = getAdminUser();
  const adminPass = readSecret("PORTAL_ADMIN_PASSWORD");
  if (adminUser === null || adminPass === null) return false;

  if (!safeEqual(user.trim(), adminUser)) return false;
  return safeEqual(pass, adminPass);
}

/** Reads and validates the session cookie. */
export function getPortalSession() {
  const cookieStore = cookies();
  const token = cookieStore.get(PORTAL_AUTH_COOKIE)?.value;
  return verifySessionToken(token);
}

/** True when a valid portal session is present. */
export function isPortalAuthenticated(): boolean {
  return getPortalSession().valid;
}

export { PORTAL_AUTH_COOKIE };
