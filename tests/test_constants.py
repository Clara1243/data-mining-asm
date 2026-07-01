from app.constants import classify_risk


def test_classify_risk_handles_non_numeric_input():
    label, color = classify_risk("invalid", 40, 70)
    assert label == "NOT APPLICABLE"
    assert color == "#6b7280"


def test_classify_risk_handles_none_input():
    label, color = classify_risk(None, 40, 70)
    assert label == "NOT APPLICABLE"
    assert color == "#6b7280"
