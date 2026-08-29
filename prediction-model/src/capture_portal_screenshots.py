from playwright.sync_api import sync_playwright

out_login = r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863\portal_login_preview.png"
out_dash = r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863\portal_dashboard_preview.png"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()

    # 1. Login Page
    page.goto("http://localhost/portal/login", wait_until="domcontentloaded")
    page.wait_for_selector("text=Data Vault", timeout=15000)
    page.wait_for_timeout(1000)
    page.screenshot(path=out_login)
    print("Saved login screenshot to:", out_login)

    # 2. Authenticate
    page.fill('input[type="password"]', "Kloudtrack2026!")
    page.click('button[type="submit"]')
    page.wait_for_selector("text=Telemetry Logs & Data Export Vault", timeout=15000)
    page.wait_for_timeout(1500)
    page.screenshot(path=out_dash)
    print("Saved dashboard screenshot to:", out_dash)

    browser.close()
