# test for pdf_text_extractor.py
from server.modules.pdf_text_extractor import extract_text_from_pdf


def test_extract_text_from_pdf(mocker):

    # Tests that the extractor iterates pages and calls get_text().

    #1. ARRANGE
    mock_page_1 = mocker.MagicMock()
    mock_page_1.get_text.return_value = "This is page one."

    mock_page_2 = mocker.MagicMock()
    mock_page_2.get_text.return_value = "This is page two."

    mock_doc = [mock_page_1, mock_page_2]

    mock_open = mocker.patch("server.modules.pdf_text_extractor.fitz.open")
    mock_open.return_value.__enter__.return_value = mock_doc

    fake_pdf_bytes = b"dummy-bytes"

    #2. ACT
    results = extract_text_from_pdf(fake_pdf_bytes)

    #3. ASSERT
    mock_open.assert_called_with(stream=fake_pdf_bytes, filetype="pdf")
    assert len(results) == 2
    assert results[0]["page"] == 1
    assert results[0]["text"] == "This is page one."
    assert results[1]["text"] == "This is page two."

    # print on success
    print(f"\n✅ PASSED: test_extract_text_from_pdf"
          f"\n   -> Test: PDF pages were iterated and text extracted correctly.")