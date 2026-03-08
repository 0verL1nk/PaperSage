from agent.application.language import detect_language


def test_detect_language_returns_zh_en_other():
    assert detect_language("这是中文文本") == "zh"
    assert detect_language("This is an english sentence") == "en"
    assert detect_language("12345 !!!") == "other"
