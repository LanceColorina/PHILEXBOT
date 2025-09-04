# pdf_chunker.py
from langchain_experimental.text_splitter import SemanticChunker
from server.modules.pii_sanitizer import PIISanitizer
from langchain_community.embeddings import HuggingFaceEmbeddings

def chunk_pdf_with_semantic(page_texts, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_method="Semantic", sanitizer: PIISanitizer = PIISanitizer()):
    """
    Splits PDF text into semantically meaningful chunks and sanitizes PII.

    Args:
        page_texts (list): [{"page": int, "text": str}, ...]
        embedding_model: LangChain-compatible embedding model (e.g., HuggingFaceEmbeddings)
        chunk_method (str): Label prefix for chunks (default: "Semantic")

    Returns:
        list of dicts with chunk_id, text, tokens, page, position_label
    """
    model = HuggingFaceEmbeddings(model_name=embedding_model)
    all_chunks = []
    chunk_id = 0

    chunker = SemanticChunker(model)

    for page_data in page_texts:
        # sanitize PII
        sanitized_text, mask_map = sanitizer.sanitize(page_data["text"])

        # split into semantic chunks
        semantic_chunks = chunker.create_documents([sanitized_text])

        for doc in semantic_chunks:
            chunk_text = doc.page_content

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,  # sanitized text
                "tokens": len(chunk_text.split()),  # rough token count
                "page": page_data["page"],
                "position_label": f"{chunk_method} {chunk_id + 1} (Page {page_data['page']})",
                "mask_map": mask_map
            })
            chunk_id += 1

    return all_chunks