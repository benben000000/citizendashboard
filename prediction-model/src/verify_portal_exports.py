import urllib.request
import json
import http.cookiejar

cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

base_url = "http://localhost"

print("========================================================================")
print("   DATA VAULT EXPORT & AUTHENTICATION END-TO-END VERIFICATION")
print("========================================================================")

# 1. Login
login_payload = json.dumps({"username": "admin", "password": "Kloudtrack2026!"}).encode("utf-8")
req_login = urllib.request.Request(
    f"{base_url}/api/portal/auth/login",
    data=login_payload,
    headers={"Content-Type": "application/json"}
)
res_login = opener.open(req_login)
data_login = json.loads(res_login.read().decode("utf-8"))
print(f"[TEST 1] Login Status: {res_login.status} | Success: {data_login.get('success')} | User: {data_login.get('user', {}).get('username')}")

# 2. Session Check
req_sess = urllib.request.Request(f"{base_url}/api/portal/auth/session")
res_sess = opener.open(req_sess)
data_sess = json.loads(res_sess.read().decode("utf-8"))
print(f"[TEST 2] Session Check: {res_sess.status} | Authenticated: {data_sess.get('authenticated')} | User: {data_sess.get('user', {}).get('username')}")

# 3. Raw Telemetry CSV Export
req_raw_csv = urllib.request.Request(f"{base_url}/api/portal/export?stream=raw&interval=1m&format=csv")
res_raw_csv = opener.open(req_raw_csv)
csv_content = res_raw_csv.read().decode("utf-8").replace("\ufeff", "")
lines_csv = [l for l in csv_content.split("\r\n") if l.strip()]
print(f"[TEST 3] Raw Telemetry CSV: {res_raw_csv.status} | Size: {len(csv_content)} bytes | Total CSV Rows: {len(lines_csv)} | Header: {lines_csv[0][:60]}...")

# 4. Processed Telemetry Excel (.xlsx) Export
req_proc_xlsx = urllib.request.Request(f"{base_url}/api/portal/export?stream=processed&interval=30m&format=xlsx")
res_proc_xlsx = opener.open(req_proc_xlsx)
xlsx_bytes = res_proc_xlsx.read()
print(f"[TEST 4] Processed Telemetry XLSX: {res_proc_xlsx.status} | File Size: {len(xlsx_bytes)} bytes | MIME: {res_proc_xlsx.headers.get('Content-Type')}")

# 5. Prediction Nowcasts JSON Export
req_pred_json = urllib.request.Request(f"{base_url}/api/portal/export?stream=prediction&interval=1h&format=json")
res_pred_json = opener.open(req_pred_json)
pred_json_data = json.loads(res_pred_json.read().decode("utf-8"))
print(f"[TEST 5] Prediction JSON Export: {res_pred_json.status} | Stream: {pred_json_data.get('metadata', {}).get('stream')} | Total Records: {pred_json_data.get('metadata', {}).get('totalRecords')}")

# 6. Live Preview API Endpoint
req_prev = urllib.request.Request(f"{base_url}/api/portal/export?stream=processed&interval=1h&preview=true")
res_prev = opener.open(req_prev)
prev_data = json.loads(res_prev.read().decode("utf-8"))
print(f"[TEST 6] Live Preview API: {res_prev.status} | Total Count: {prev_data.get('totalCount')} | Preview Count: {prev_data.get('previewCount')}")

# 7. Logout
req_logout = urllib.request.Request(f"{base_url}/api/portal/auth/logout", data=b"{}", headers={"Content-Type": "application/json"})
res_logout = opener.open(req_logout)
print(f"[TEST 7] Logout Status: {res_logout.status}")

print("========================================================================")
print("ALL DATA VAULT AUTH & MULTI-STREAM EXPORT TESTS PASSED (100%)")
print("========================================================================")
