from pathlib import Path
from unittest.mock import MagicMock, patch


def test_detect_and_store_calls_download_and_load():
    fake_combos = [{"id": "1-2", "status": "OK", "uses": []}]

    with patch("combos.detector.asyncio.run", return_value=fake_combos), \
         patch("combos.detector.load_combos", return_value=42) as mock_load:
        from combos.detector import detect_and_store
        count = detect_and_store()

    assert count == 42
    mock_load.assert_called_once_with(fake_combos)


def test_detect_and_store_returns_load_count():
    fake_combos = [{"id": "1-2", "status": "OK", "uses": []}]

    with patch("combos.detector.asyncio.run", return_value=fake_combos), \
         patch("combos.detector.load_combos", return_value=99):
        from combos.detector import detect_and_store
        count = detect_and_store()

    assert count == 99


def test_detect_and_store_applies_limit():
    fake_combos = [{"id": str(i), "status": "OK", "uses": []} for i in range(5)]

    with patch("combos.detector.asyncio.run", return_value=fake_combos), \
         patch("combos.detector.load_combos", return_value=3) as mock_load:
        from combos.detector import detect_and_store
        detect_and_store(limit=3)

    passed = mock_load.call_args[0][0]
    assert len(passed) == 3


def test_detect_and_store_accepts_cache_path():
    fake_combos = []

    with patch("combos.detector.asyncio.run", return_value=fake_combos), \
         patch("combos.detector.load_combos", return_value=0):
        from combos.detector import detect_and_store
        count = detect_and_store(limit=0, cache_path=Path("data/combos.json"))

    assert count == 0


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
