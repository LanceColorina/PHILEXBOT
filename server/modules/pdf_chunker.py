import re

def manual_semantic_chunker(text: str, max_words: int = 100):
    """
    Splits a large text into roughly semantic chunks by grouping sentences
    until a word limit is reached.
    """

    # Split into sentences using punctuation as simple boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If adding this sentence exceeds max_words, start a new chunk
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    # Add remaining sentences as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_pdf_with_semantic(page_texts, chunk_method="ManualSemantic", max_words=100):
    """
    Splits PDF text into manually created semantic-like chunks.
    
    Args:
        page_texts (list): [{"page": int, "text": str}, ...]
        chunk_method (str): Label prefix for chunks
        max_words (int): Maximum words per chunk

    Returns:
        list of dicts with chunk_id, text, tokens, page, position_label
    """
    all_chunks = []
    chunk_id = 0

    for page_data in page_texts:
        text = page_data["text"]
        semantic_chunks = manual_semantic_chunker(text, max_words=max_words)

        for i, chunk_text in enumerate(semantic_chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "tokens": len(chunk_text.split()),
                "page": page_data["page"],
                "position_label": f"{chunk_method} {chunk_id + 1} (Page {page_data['page']})",
            })
            chunk_id += 1

    return all_chunks
