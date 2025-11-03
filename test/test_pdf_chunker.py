# test for pdf_chunker.py
from server.modules.pdf_chunker import manual_semantic_chunker, chunk_pdf_with_semantic


def test_manual_semantic_chunker():
    # Tests the core chunking logic.
    text = ("This is sentence one. This is sentence two. "
            "This is sentence three. This is sentence four. "
            "This is sentence five.")

    chunks = manual_semantic_chunker(text, max_words=10)

    assert len(chunks) == 3
    assert chunks[0] == "This is sentence one. This is sentence two."
    assert chunks[2] == "This is sentence five."

    # This will now print on success!
    print(f"\n✅ PASSED: test_manual_semantic_chunker"
          f"\n   -> Test: Text was split into the correct 3 chunks.")


def test_chunk_pdf_with_semantic():
    # Tests the wrapper function that adds metadata.

    dummy_pages = [
        {"page": 1, "text": "Page one, sentence one. Page one, sentence two."},
        {"page": 2, "text": "Page two, sentence one."}
    ]

    chunks = chunk_pdf_with_semantic(dummy_pages, max_words=5)

    assert len(chunks) == 3
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_id"] == 0
    assert chunks[2]["page"] == 2
    assert chunks[2]["chunk_id"] == 2

    # print on success
    print(f"\n✅ PASSED: test_chunk_pdf_with_semantic"
          f"\n   -> Test: Page data was chunked with correct metadata (IDs, page numbers).")