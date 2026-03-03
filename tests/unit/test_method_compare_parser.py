from utils.compare_parser import method_compare_to_csv, parse_method_compare_payload


def test_parse_method_compare_payload_from_json_block():
    raw = """
    这里是前缀
    {
      "topic": "A vs B",
      "columns": ["维度", "方法A", "方法B"],
      "rows": [
        {"维度": "目标", "方法A": "高精度", "方法B": "低延迟"},
        {"维度": "成本", "方法A": "较高", "方法B": "较低"}
      ],
      "recommendation": "按延迟优先选择 B"
    }
    """
    payload = parse_method_compare_payload(raw)

    assert payload is not None
    assert payload["topic"] == "A vs B"
    assert len(payload["rows"]) == 2
    assert payload["columns"] == ["维度", "方法A", "方法B"]


def test_parse_method_compare_payload_returns_none_on_invalid():
    assert parse_method_compare_payload("not-json") is None
    assert parse_method_compare_payload('{"rows":"bad"}') is None


def test_method_compare_to_csv_contains_headers_and_rows():
    payload = {
        "columns": ["Dimension", "MethodA", "MethodB"],
        "rows": [{"Dimension": "Goal", "MethodA": "Acc", "MethodB": "Speed"}],
    }
    csv_text = method_compare_to_csv(payload)

    assert "Dimension,MethodA,MethodB" in csv_text
    assert "Goal,Acc,Speed" in csv_text
