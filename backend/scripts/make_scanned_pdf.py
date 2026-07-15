"""
Create a scanned-style PDF (image page, no text layer) for OCR verification.

Usage (from backend/ with venv active):
  python scripts/make_scanned_pdf.py
→ writes ../sample_scanned_handbook.pdf
"""

from pathlib import Path

import fitz


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "sample_scanned_handbook.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    text = (
        "Nimbus Cloud Storage — Scanned Policy Memo\n\n"
        "Confidential OCR verification document.\n\n"
        "Effective 1 January 2026, all remote employees must complete\n"
        "the annual security quiz within 14 calendar days of hire.\n\n"
        "The unique passphrase for this memo is BLUE-ORBIT-77.\n\n"
        "Facilities in Bangalore observe a silent hour from 14:00 to 15:00 IST."
    )

    # Draw text onto a pixmap-like page by inserting as text, then converting
    # the whole page to an image-only PDF (removes text layer).
    page.insert_text((50, 72), text, fontsize=14, fontname="helv")

    # Rasterize and rebuild as image-only page
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img_pdf = fitz.open()
    img_page = img_pdf.new_page(width=page.rect.width, height=page.rect.height)
    img_page.insert_image(img_page.rect, pixmap=pix)

    img_pdf.save(out)
    img_pdf.close()
    doc.close()

    # Prove there is no extractable text layer
    check = fitz.open(out)
    native = (check[0].get_text() or "").strip()
    check.close()
    print(f"Wrote {out}")
    print(f"Native text layer chars: {len(native)} (expect 0 for a true scan)")


if __name__ == "__main__":
    main()
