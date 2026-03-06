from agent.ui_helpers import _infer_mindmap_iframe_height


def test_infer_mindmap_iframe_height_from_embedded_chart_height():
    html = "<style>#mindmap{width:100%;height:520px;}</style>"
    assert _infer_mindmap_iframe_height(html) == 680


def test_infer_mindmap_iframe_height_uses_default_when_height_missing():
    assert _infer_mindmap_iframe_height("<html><body>no-style</body></html>") == 660
