def embed_texts(chunks: list[dict], model) -> list[dict]:
    """
    Adds embeddings to each chunk and returns the updated list.

    Args:
        chunks (list[dict]): List of chunk dicts, each with at least "text".
        model (str): HuggingFace embedding model to use.

    Returns:
        list[dict]: Chunks with an added "embedding" key.
    """
    

    for c in chunks:
        c["embedding"] = model.embed_query(c["text"])

    return chunks