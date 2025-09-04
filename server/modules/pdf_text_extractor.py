import fitz  # PyMuPDF

def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """
    Extracts text from a PDF file.

    Args:
        file_bytes (bytes): The binary content of the PDF file.

    Returns:
        list[dict]: A list of dictionaries with keys:
            - "page": Page number (1-indexed)
            - "text": Extracted text content from that page
    """
    page_texts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            page_texts.append({
                "page": i + 1,
                "text": page.get_text()
            })
    return page_texts