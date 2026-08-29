import os
import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def compile_pdf():
    html_path = os.path.abspath(r"c:\Ben File\beta-citizen-prediction\prediction-model\docs\Garcia_PINN_LNN_Working_Paper.html")
    pdf_path = os.path.abspath(r"c:\Ben File\beta-citizen-prediction\prediction-model\docs\Garcia_PINN_LNN_Working_Paper.pdf")
    
    print("=" * 80)
    print("📄 COMPILING ACADEMIC WORKING PAPER PDF VIA PLAYWRIGHT CHROMIUM")
    print(f"👉 Input HTML: {html_path}")
    print(f"👉 Target PDF: {pdf_path}")
    print("=" * 80)

    if not os.path.exists(html_path):
        print(f"❌ Error: HTML file not found at {html_path}")
        return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to local file URI
        file_url = f"file:///{html_path.replace(os.sep, '/')}"
        print(f"🌐 Loading page: {file_url}...")
        page.goto(file_url, wait_until="networkidle")
        
        # Wait for KaTeX and Google Fonts to render
        time.sleep(3)
        
        print("🖨️ Rendering vector PDF...")
        page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={
                "top": "18mm",
                "bottom": "18mm",
                "left": "16mm",
                "right": "16mm"
            },
            display_header_footer=True,
            header_template='<div style="font-size: 7.5pt; font-family: sans-serif; color: #94a3b8; width: 100%; text-align: right; padding-right: 16mm; font-style: italic;">Garcia PINN-LNN Working Paper (2026)</div>',
            footer_template='<div style="font-size: 8pt; font-family: sans-serif; color: #64748b; width: 100%; text-align: center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
        )
        
        browser.close()

    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"\n🎉 SUCCESS! Working Paper PDF compiled successfully: {pdf_path} ({size_kb:.1f} KB)")
        return True
    else:
        print("\n❌ PDF compilation failed.")
        return False

if __name__ == "__main__":
    compile_pdf()
