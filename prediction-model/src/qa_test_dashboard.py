import os
import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ARTIFACTS_DIR = r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863"
SCREENSHOT_PATH = os.path.join(ARTIFACTS_DIR, "prediction_dashboard_qa.png")

def test_prediction_ui():
    print("🚀 Launching Playwright Chromium to QA Prediction Dashboard...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("🌐 Navigating to http://localhost:80/prediction...")
        page.goto("http://localhost:80/prediction", wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # Verify Key Elements
        title = page.title()
        print(f"  ✓ Page Title: '{title}'")

        # Check badges
        pinn_badge = page.locator("text=PINN-LNN Continuous Neural ODE").first
        if pinn_badge.is_visible():
            print("  ✓ PINN-LNN Continuous Neural ODE badge is VISIBLE!")
        else:
            print("  ⚠️ PINN-LNN badge not detected.")

        # Check temperature element
        temp_elem = page.locator("text=°C").first
        if temp_elem.is_visible():
            print("  ✓ Live Temperature display is VISIBLE!")

        # Take full page screenshot
        print(f"📸 Capturing high-res screenshot to: {SCREENSHOT_PATH}")
        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        print("🎉 QA Screenshot captured successfully!")

        browser.close()

if __name__ == "__main__":
    test_prediction_ui()
