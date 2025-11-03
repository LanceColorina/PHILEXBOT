# test for answer_generation.py
from server.modules.answer_generation import generate_answer


def test_generate_answer_formatting(mocker):
    #Tests that the function correctly calls the generator and formats the output.
    #1. ARRANGE
    context = "The Constitution of the Philippines was ratified in 1987."

    # context[52:56] == "1987"
    fake_result = {
        "answer": "1987",
        "start": 52,  # Corrected start index
        "end": 56  # Corrected end index
    }

    mock_generator = mocker.patch("server.modules.answer_generation.generator")
    mock_generator.return_value = fake_result

    question = "When was the Constitution ratified?"

    #2. ACT
    result = generate_answer(question, context)

    #3. ASSERT
    expected_output = "Answer: 1987\n\nContext Snippet: 1987"
    assert result == expected_output

    # print on success
    print(f"\n✅ PASSED: test_generate_answer_formatting"
          f"\n   -> Test: Formatted answer matches expected output.")