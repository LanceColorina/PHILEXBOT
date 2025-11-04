from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(question: str, context: str, results=None) -> str:
    """
    Generates an answer based on the question and context using a small text generation model.
    Optionally includes page numbers from Qdrant query results.

    Args:
        question (str): The user's question.
        context (str): The context text to base the answer on.
        results (list, optional): List of Qdrant ScoredPoint objects or dicts.

    Returns:
        str: The generated answer (and pages if results are provided).
    """
    # Build prompt
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    result = generator(prompt, max_new_tokens=150, temperature=0.7)
    answer = result[0]["generated_text"].strip()

    # --- Determine which pages contributed to the context ---
    used_pages = set()
    if results:
        for r in results:
            try:
                payload = r.payload if hasattr(r, "payload") else r.get("payload", {})
                page = payload.get("page")
                text_chunk = payload.get("text", "")
                # Check if this chunk's text appears in the context
                if page is not None and text_chunk and text_chunk.strip()[:30] in context:
                    used_pages.add(page)
            except Exception:
                continue

    # --- Build return output ---
    page_info = ""
    if used_pages:
        page_info = f"\n\n📄 Used page(s): {', '.join(map(str, sorted(used_pages)))}"
    else:
        page_info = "\n\n📄 No matching pages found in context."

    return f"Answer: {answer}{page_info}"

def main():
    print("🧠 Module Test: generate_answer.py (google/flan-t5-base\n")

    # Example context and question
    context = (
        "The Constitution of the Philippines was ratified in 1987. "
        "It serves as the supreme law of the country and defines the structure "
        "of government, the rights of citizens, and the guiding principles of the nation."
    )

    question = "When was the Constitution of the Philippines ratified?"

    print(f"Question: {question}\n")
    print(f"Context: {context}\n")

    # Run QA
    result = generate_answer(question, context)

    print("=== Result ===")
    print(result)


if __name__ == '__main__':
    main()