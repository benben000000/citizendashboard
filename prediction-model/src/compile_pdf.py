"""
Compile Garcia_PINN_LNN_Working_Paper.html to high-resolution vector PDF using Microsoft Edge Chromium headless.
"""

import os
import sys
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def compile_pdf():
    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "Garcia_PINN_LNN_Working_Paper.html"))
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "Garcia_PINN_LNN_Working_Paper.pdf"))
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    print("=" * 80)
    print("📄 COMPILING ACADEMIC WORKING PAPER PDF VIA CHROMIUM")
    print(f"👉 Input HTML: {html_path}")
    print(f"👉 Target PDF: {pdf_path}")
    print("=" * 80)

    if not os.path.exists(html_path):
        print(f"❌ Error: HTML file not found at {html_path}")
        return False

    url_path = f"file:///{html_path.replace(os.sep, '/')}"
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        url_path
    ]
    
    subprocess.run(cmd, check=True)
    
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"🎉 SUCCESS: Vector PDF compiled successfully: {pdf_path} ({size_kb:.1f} KB)")
        return True
    else:
        print("❌ Error: PDF output file was not created.")
        return False

if __name__ == "__main__":
    compile_pdf()
