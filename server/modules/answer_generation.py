from transformers import pipeline

generator = pipeline("question-answering", model="deepset/roberta-base-squad2")

def generate_answer(question: str, context: str) -> str:
    """
    Generates an answer based on the question and context using a text generation model.

    Args:
        question (str): The user's question.
        context (str): The context to base the answer on.

    Returns:
        str: The generated answer.
    """
    print(context)
    result = generator(question=question, context=context)
    answer = result["answer"]
    start, end = result["start"], result["end"]
    snippet = context[start:end]
    return "Answer: " + answer + "\n\nContext Snippet: " + snippet

def main():
    print("🧠 Module Test: generate_answer.py (MiniLM-L3-v2 + RoBERTa-SQuAD2)\n")

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