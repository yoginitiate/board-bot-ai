from board_bot.utils.safety import mask_pii, risk_score_rule


def test_mask_pii() -> None:
    text = "연락처 010-1234-5678, 이메일 a@test.com"
    masked = mask_pii(text)
    assert "[PHONE]" in masked
    assert "[EMAIL]" in masked


def test_risk_high() -> None:
    out = risk_score_rule("법적 조치 하겠습니다")
    assert out["level"] == "HIGH"
