# test for embed_text.py
from server.modules.embed_text import embed_texts


def test_embed_texts_adds_embedding_key():
    # Tests that the 'embedding' key is correctly added to each chunk.

    #1. ARRANGE
    class MockEmbeddingModel:
        def embed_query(self, text):
            return [0.1] * len(text.split())

    mock_model = MockEmbeddingModel()
    input_chunks = [
        {"text": "hello world"},
        {"text": "this is a test"}
    ]

    #2. ACT
    output_chunks = embed_texts(input_chunks, mock_model)

    #3. ASSERT
    assert "embedding" in output_chunks[0]
    assert output_chunks[0]["embedding"] == [0.1, 0.1]
    assert output_chunks[1]["embedding"] == [0.1, 0.1, 0.1, 0.1]

    # print on success
    print(f"\n✅ PASSED: test_embed_texts_adds_embedding_key"
          f"\n   -> Test: 'embedding' key was added and calculated correctly.")