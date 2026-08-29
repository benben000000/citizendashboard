"""
Compile Garcia_PINN_LNN_Working_Paper.html to high-resolution vector PDF using Chromium.
"""

import os
from playwright.sync_api import sync_playwright

def compile_pdf():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        html_path = os.path.abspath('prediction-model/docs/Garcia_PINN_LNN_Working_Paper.html')
        pdf_path = os.path.abspath('prediction-model/docs/Garcia_PINN_LNN_Working_Paper.pdf')
        
        url_path = html_path.replace('\\', '/')
        page.goto(f'file:///{url_path}', wait_until='networkidle')
        page.wait_for_timeout(2500)
        
        page.pdf(
            path=pdf_path,
            format='A4',
            print_background=True,
            margin={
                'top': '12mm',
                'bottom': '12mm',
                'left': '14mm',
                'right': '14mm'
            }
        )
        size = os.path.getsize(pdf_path)
        print(f"Successfully compiled vector PDF: {pdf_path} ({size / 1024:.1f} KB)")
        browser.close()

if __name__ == '__main__':
    compile_pdf()
