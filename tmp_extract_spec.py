from pathlib import Path
try:
    from pypdf import PdfReader
    pdf_path = Path(r"C:\Users\Javoh\AppData\Local\Packages\38833FF26BA1D.UnigramPreview_g9c9v27vpyspw\LocalState\0\documents\RAG API   test spetsifikatsiyasi (lex.uz).pdf")
    print('exists:', pdf_path.exists())
    reader = PdfReader(str(pdf_path))
    print('pages:', len(reader.pages))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(text[:20000])
except Exception:
    import traceback; traceback.print_exc()
