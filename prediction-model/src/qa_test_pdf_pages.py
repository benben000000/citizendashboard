import os
import sys
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def capture_paper_screenshots():
    html_path = os.path.abspath(r"c:\Ben File\beta-citizen-prediction\prediction-model\docs\Garcia_PINN_LNN_Working_Paper.html")
    shot1_path = os.path.abspath(r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863\working_paper_page1_preview.png")
    shot2_path = os.path.abspath(r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863\working_paper_page2_preview.png")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 1600})
        
        file_url = f"file:///{html_path.replace(os.sep, '/')}"
        page.goto(file_url, wait_until="networkidle")
        page.wait_for_timeout(2000)

        # Full page screenshot
        page.screenshot(path=shot1_path, full_page=False)
        print(f"📸 Page 1 Screenshot: {shot1_path}")

        # Scroll down to tables and diagrams
        page.evaluate("window.scrollTo(0, 1600)")
        page.wait_for_timeout(1000)
        page.screenshot(path=shot2_path, full_page=False)
        print(f"📸 Page 2 Screenshot: {shot2_path}")

        # Page 3 (Tables & References)
        shot3_path = os.path.abspath(r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863\working_paper_page3_preview.png")
        page.evaluate("window.scrollTo(0, 3200)")
        page.wait_for_timeout(1000)
        page.screenshot(path=shot3_path, full_page=False)
        print(f"📸 Page 3 Screenshot: {shot3_path}")

        browser.close()

if __name__ == "__main__":
    capture_paper_screenshots()
