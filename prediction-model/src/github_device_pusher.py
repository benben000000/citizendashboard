import os
import sys
import time
import json
import urllib.request
import urllib.parse
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CLIENT_ID = "178c6fc778ccc68e1d6a"  # Git Credential Manager official Client ID
REPO_URL = "https://github.com/benben000000/citizendashboard.git"
STATUS_FILE = os.path.join(os.path.dirname(__file__), "device_auth_status.json")

# Ensure git is in PATH
git_cmd = r"C:\Users\Kloudtech Software\AppData\Local\MinGit\cmd"
git_bin = r"C:\Users\Kloudtech Software\AppData\Local\MinGit\mingw64\bin"
os.environ["PATH"] = f"{git_cmd};{git_bin};" + os.environ.get("PATH", "")

def start_device_flow():
    # 1. Request device code
    data = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "scope": "repo,read:user,user:email"
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://github.com/login/device/code",
        data=data,
        headers={"Accept": "application/json"}
    )
    
    with urllib.request.urlopen(req) as resp:
        device_data = json.loads(resp.read().decode("utf-8"))

    device_code = device_data["device_code"]
    user_code = device_data["user_code"]
    verification_uri = device_data["verification_uri"]
    interval = device_data.get("interval", 5)
    expires_in = device_data.get("expires_in", 900)

    # Save to status file
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "status": "WAITING_FOR_USER",
            "user_code": user_code,
            "verification_uri": verification_uri,
            "expires_in": expires_in,
            "started_at": time.time(),
        }, f, indent=2)

    print("=" * 70, flush=True)
    print("🔑 GITHUB 1-CLICK DEVICE AUTHORIZATION", flush=True)
    print(f"👉 1. Open: {verification_uri}", flush=True)
    print(f"👉 2. Enter Code: {user_code}", flush=True)
    print(f"👉 3. Click 'Authorize'", flush=True)
    print("=" * 70, flush=True)
    print("⏳ Waiting for authorization...", flush=True)

    # 2. Poll for token
    token_url = "https://github.com/login/oauth/access_token"
    start_time = time.time()

    while time.time() - start_time < expires_in:
        time.sleep(interval)
        poll_data = urllib.parse.urlencode({
            "client_id": CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
        }).encode("utf-8")

        poll_req = urllib.request.Request(
            token_url,
            data=poll_data,
            headers={"Accept": "application/json"}
        )

        try:
            with urllib.request.urlopen(poll_req) as token_resp:
                res = json.loads(token_resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Poll request error: {e}", flush=True)
            continue

        if "access_token" in res:
            access_token = res["access_token"]
            print("\n🎉 Authorization received successfully!", flush=True)
            
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump({"status": "AUTHORIZED"}, f, indent=2)

            # 3. Perform Git Push
            auth_repo_url = f"https://x-access-token:{access_token}@github.com/benben000000/citizendashboard.git"
            print("🚀 Pushing local commits to origin main...", flush=True)
            
            push_res = subprocess.run(
                ["git", "push", auth_repo_url, "main:main"],
                capture_output=True,
                text=True
            )
            print("STDOUT:", push_res.stdout, flush=True)
            print("STDERR:", push_res.stderr, flush=True)

            if push_res.returncode == 0:
                print("✅ SUCCESS! Commits have been pushed to GitHub!", flush=True)
                with open(STATUS_FILE, "w", encoding="utf-8") as f:
                    json.dump({"status": "PUSH_SUCCESS"}, f, indent=2)
                return True
            else:
                print(f"❌ Git push failed with exit code: {push_res.returncode}", flush=True)
                with open(STATUS_FILE, "w", encoding="utf-8") as f:
                    json.dump({"status": "PUSH_FAILED", "error": push_res.stderr}, f, indent=2)
                return False

        error = res.get("error")
        if error == "authorization_pending":
            continue
        elif error == "slow_down":
            interval += 5
        elif error in ["expired_token", "access_denied"]:
            print(f"❌ Authentication ended: {error}", flush=True)
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump({"status": "AUTH_DENIED_OR_EXPIRED"}, f, indent=2)
            return False

    print("❌ Timed out waiting for device authorization.", flush=True)
    return False

if __name__ == "__main__":
    start_device_flow()
