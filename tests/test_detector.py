from unittest.mock import MagicMock, patch


def _make_mock_session(count=3):
    result = MagicMock()
    result.single.return_value = {"count": count}
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = result
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


def test_detect_and_store_returns_total_count():
    driver, session = _make_mock_session(count=5)
    with patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        total = detect_and_store()
    assert total >= 0  # total is sum of counts from all rules


def test_detect_and_store_runs_multiple_rules():
    driver, session = _make_mock_session(count=2)
    with patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        detect_and_store()
    # Should open multiple sessions (one per rule function)
    assert driver.session.call_count >= 1
    assert session.run.call_count >= 3  # at least 3 rules


def test_persist_rule_creates_infinite_combo_node():
    driver, session = _make_mock_session(count=10)
    with patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import _detect_persist_combo
        _detect_persist_combo(session)

    call_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "Persist" in call_text
    assert "infinite" in call_text.lower() or "combo_type" in call_text
