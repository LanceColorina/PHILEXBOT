# test for pii_sanitizer.py
from server.modules.pii_sanitizer import PIISanitizer


def test_pii_sanitizer_logic(mocker):
    #Tests the custom masking and restoring logic.
    # 1.ARRANGE
    fake_result = mocker.Mock(entity_type="PERSON", start=11, end=19)  # "John Doe"

    mock_analyze = mocker.patch("server.modules.pii_sanitizer.AnalyzerEngine.analyze")
    mock_analyze.return_value = [fake_result]

    sanitizer = PIISanitizer()
    text = "My name is John Doe."

    # 2.ACT (Sanitize)
    masked, mask_map = sanitizer.sanitize(text)

    # 3.ASSERT (Sanitize again)
    mock_analyze.assert_called_with(text=text, language="en")
    assert masked == "My name is [PERSON_1]."
    assert mask_map == {"PERSON_1": "John Doe"}

    # 4. ACT (Restore)
    restored = sanitizer.restore(masked)

    # 5. ASSERT (Restore)
    assert restored == text

    # print on success
    print(f"\n✅ PASSED: test_pii_sanitizer_logic"
          f"\n   -> Test: Text was correctly masked and then restored.")