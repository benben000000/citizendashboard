import os
import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ARTIFACTS_DIR = r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863"

def test_multi_horizon():
    print("🚀 Running Multi-Horizon Interactive Playwright Test...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.goto("http://localhost:80/prediction", wait_until="networkidle", timeout=30000)
        time.sleep(2)

        horizons = ["1h", "3h", "6h", "12h", "24h"]
        for h in horizons:
            # Click horizon button
            btn = page.locator(f"button:has-text('{h}')").first
            btn.click()
            time.sleep(1.5)

            # Get displayed temperature text
            temp_text = page.locator("p.tabular-nums").first.inner_text().strip()
            # Get displayed sub-label time
            time_elem = page.locator("p:has-text('Temperatura')").first.inner_text().strip()
            print(f"  👉 Horizon: {h:<4} | Displayed Temp: {temp_text}°C | Label: {time_elem}")

        # Capture 1h screenshot
        page.locator("button:has-text('1h')").first.click()
        time.sleep(1.5)
        path_1h = os.path.join(ARTIFACTS_DIR, "prediction_1h_qa.png")
        page.screenshot(path=path_1h, full_page=True)
        print(f"📸 Captured 1h Horizon Screenshot: {path_1h}")

        # Capture 3h screenshot
        page.locator("button:has-text('3h')").first.click()
        time.sleep(1.5)
        path_3h = os.path.join(ARTIFACTS_DIR, "prediction_3h_qa.png")
        page.screenshot(path=path_3h, full_page=True)
        print(f"📸 Captured 3h Horizon Screenshot: {path_3h}")

        browser.close()

if __name__ == "__main__":
    test_multi_horizon()
