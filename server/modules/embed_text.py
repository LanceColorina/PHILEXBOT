from langchain_community.embeddings import HuggingFaceEmbeddings

def embed_texts(chunks: list[dict], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[dict]:
    """
    Adds embeddings to each chunk and returns the updated list.

    Args:
        chunks (list[dict]): List of chunk dicts, each with at least "text".
        embedding_model (str): HuggingFace embedding model to use.

    Returns:
        list[dict]: Chunks with an added "embedding" key.
    """
    model = HuggingFaceEmbeddings(model_name=embedding_model)

    for c in chunks:
        c["embedding"] = model.embed_query(c["text"])

    return chunks