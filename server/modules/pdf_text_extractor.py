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


def main():
    # Ask user for PDF filename
    file_name = input("Enter the PDF file name (e.g. sample.pdf): ").strip()

    try:
        # Read the file in binary mode
        with open(file_name, "rb") as f:
            file_bytes = f.read()

        # Extract text from the PDF
        results = extract_text_from_pdf(file_bytes)

        # Print summary of extracted content
        print(f"\n✅ Extracted text from {len(results)} pages.\n")
        for page_data in results:
            print(f"--- Page {page_data['page']} ---")
            print(page_data['text'][:500] + ('...' if len(page_data['text']) > 500 else ''))
            print()

    except FileNotFoundError:
        print("❌ File not found. Please make sure the file exists in this directory.")
    except Exception as e:
        print(f"⚠️ An error occurred: {e}")

if __name__ == '__main__':
    main()