from unittest.mock import MagicMock, patch


def _make_mock_driver():
    tx = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.execute_write = MagicMock(side_effect=lambda fn, *args, **kwargs: fn(tx, *args, **kwargs))
    driver = MagicMock()
    driver.session.return_value = session
    return driver, tx


def test_parse_card_extracts_all_fields():
    from ingestion.loader import parse_card
    raw = {
        "id": "abc123",
        "name": "Lightning Bolt",
        "oracle_text": "Deal 3 damage to any target.",
        "mana_cost": "{R}",
        "cmc": 1.0,
        "type_line": "Instant",
        "rarity": "common",
        "colors": ["R"],
        "keywords": [],
    }
    result = parse_card(raw)
    assert result["id"] == "abc123"
    assert result["name"] == "Lightning Bolt"
    assert result["oracle_text"] == "Deal 3 damage to any target."
    assert result["mana_cost"] == "{R}"
    assert result["cmc"] == 1.0
    assert result["colors"] == ["R"]
    assert result["keywords"] == []


def test_parse_card_handles_missing_optional_fields():
    from ingestion.loader import parse_card
    raw = {"id": "xyz", "name": "Bare Card"}
    result = parse_card(raw)
    assert result["oracle_text"] == ""
    assert result["mana_cost"] == ""
    assert result["keywords"] == []
    assert result["colors"] == []
    assert result["cmc"] == 0.0


def test_load_cards_skips_cards_without_id():
    from ingestion.loader import load_cards
    driver, tx = _make_mock_driver()
    cards = [
        {"id": "valid1", "name": "Card A", "keywords": [], "colors": [], "type_line": "Creature"},
        {"name": "No ID card"},  # should be skipped
    ]
    with patch("ingestion.loader.get_driver", return_value=driver):
        load_cards(cards)

    # execute_write should have been called (cards with id loaded)
    assert driver.session.called


def test_load_cards_calls_write_for_each_aspect():
    from ingestion.loader import load_cards
    driver, tx = _make_mock_driver()
    cards = [{"id": "c1", "name": "Card A", "keywords": ["Flying"], "colors": ["W"], "type_line": "Creature — Angel"}]

    with patch("ingestion.loader.get_driver", return_value=driver):
        load_cards(cards)

    # Should call execute_write 4 times: cards, keywords, types, colors
    assert driver.session.return_value.__enter__.return_value.execute_write.call_count == 4
