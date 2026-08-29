import os
import sys
import pypdfium2 as pdfium

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def render_all_pdf_pages():
    pdf_path = os.path.abspath(r"c:\Ben File\beta-citizen-prediction\prediction-model\docs\Garcia_PINN_LNN_Working_Paper.pdf")
    output_dir = os.path.abspath(r"C:\Users\Kloudtech Software\.gemini\antigravity-ide\brain\b7cac2c6-c55a-493c-aab8-ef053f53d863")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return

    pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(pdf)
    print(f"📄 Total PDF Pages: {total_pages}")

    for i, page in enumerate(pdf):
        page_num = i + 1
        bitmap = page.render(scale=2.0)  # 150 DPI render
        pil_image = bitmap.to_pil()
        out_file = os.path.join(output_dir, f"working_paper_page{page_num}_preview.png")
        pil_image.save(out_file)
        print(f"📸 Rendered Page {page_num}/{total_pages} -> {out_file}")

if __name__ == "__main__":
    render_all_pdf_pages()
