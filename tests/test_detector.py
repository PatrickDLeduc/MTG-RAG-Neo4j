from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_keyword_session(count_per_synergy: int = 0):
    """Return a mock Neo4j session whose run().single() returns a given count."""
    result_mock = MagicMock()
    result_mock.single.return_value = {"count": count_per_synergy}
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = result_mock
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


def test_detect_and_store_calls_download_and_load():
    fake_combos = [{"id": "1-2", "status": "OK", "uses": []}]
    driver, _ = _make_keyword_session(0)

    with patch("combos.detector.asyncio.run", return_value=(fake_combos, False)), \
         patch("combos.detector.load_combos", return_value=42) as mock_load, \
         patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        count, from_cache = detect_and_store()

    assert count == 42
    assert from_cache is False
    mock_load.assert_called_once_with(fake_combos)


def test_detect_and_store_returns_load_count():
    fake_combos = [{"id": "1-2", "status": "OK", "uses": []}]
    driver, _ = _make_keyword_session(0)

    with patch("combos.detector.asyncio.run", return_value=(fake_combos, True)), \
         patch("combos.detector.load_combos", return_value=99), \
         patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        count, from_cache = detect_and_store()

    assert count == 99
    assert from_cache is True


def test_detect_and_store_applies_limit():
    fake_combos = [{"id": str(i), "status": "OK", "uses": []} for i in range(5)]
    driver, _ = _make_keyword_session(0)

    with patch("combos.detector.asyncio.run", return_value=(fake_combos, False)), \
         patch("combos.detector.load_combos", return_value=3) as mock_load, \
         patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        detect_and_store(limit=3)

    passed = mock_load.call_args[0][0]
    assert len(passed) == 3


def test_detect_and_store_accepts_cache_path():
    fake_combos = []
    driver, _ = _make_keyword_session(0)

    with patch("combos.detector.asyncio.run", return_value=(fake_combos, True)), \
         patch("combos.detector.load_combos", return_value=0), \
         patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        count, _ = detect_and_store(limit=0, cache_path=Path("data/combos.json"))

    assert count == 0


def test_detect_and_store_includes_keyword_synergy_count():
    """Keyword synergies are counted in the total returned."""
    fake_combos = []
    driver, _ = _make_keyword_session(count_per_synergy=5)  # 3 synergies x 5 = 15

    with patch("combos.detector.asyncio.run", return_value=(fake_combos, False)), \
         patch("combos.detector.load_combos", return_value=10), \
         patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        count, _ = detect_and_store()

    assert count == 10 + 15  # 10 spellbook + 15 keyword synergies


def test_detect_and_store_calls_keyword_synergies():
    """detect_and_store must run keyword synergy detection, not just the API import."""
    fake_combos = []
    driver, session = _make_keyword_session(0)

    with patch("combos.detector.asyncio.run", return_value=(fake_combos, False)), \
         patch("combos.detector.load_combos", return_value=0), \
         patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        detect_and_store()

    # _detect_keyword_synergies runs 3 Cypher queries (one per synergy type)
    assert session.run.call_count >= 3


def test_keyword_synergies_fallback_creates_combo_nodes():
    """The old hardcoded rules still work as a private fallback."""
    result_mock = MagicMock()
    result_mock.single.return_value = {"count": 3}
    session = MagicMock()
    session.run.return_value = result_mock

    from combos.detector import _detect_keyword_synergies
    _detect_keyword_synergies(session)

    assert session.run.call_count >= 3
    calls_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "Deathtouch" in calls_text or "Persist" in calls_text
