import os
import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ARTIFACTS_DIR = r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863"
SCREENSHOT_CALUMPIT = os.path.join(ARTIFACTS_DIR, "prediction_calumpit_qa.png")

def test_station_switching():
    print("🚀 Running Playwright Station Switching Test...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Test Calumpit direct URL
        page.goto("http://localhost:80/prediction?location=Calumpit", wait_until="networkidle", timeout=30000)
        time.sleep(2)

        print("📸 Capturing Calumpit Station QA Screenshot...")
        page.screenshot(path=SCREENSHOT_CALUMPIT, full_page=True)
        print("🎉 Calumpit QA Screenshot captured successfully!")

        browser.close()

if __name__ == "__main__":
    test_station_switching()
