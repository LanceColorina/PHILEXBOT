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

    result = generator(question=question, context=context)
    answer = result["answer"]
    start, end = result["start"], result["end"]
    snippet = context[start:end]
    return "Answer: " + answer + "\n\nContext Snippet: " + snippet