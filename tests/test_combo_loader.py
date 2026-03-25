from unittest.mock import MagicMock, patch


FAKE_VARIANT = {
    "id": "742-1295",
    "uses": [
        {"card": {"name": "Kitchen Finks"}},
        {"card": {"name": "Melira, Sylvok Outcast"}},
    ],
    "identity": "B,G",
    "popularity": 500,
    "description": "Infinite life gain. Sacrifice Kitchen Finks to a sac outlet.",
    "status": "OK",
}

FAKE_VARIANT_NULLS = {
    "id": "9-9",
    "uses": [],
    "identity": None,
    "popularity": None,
    "description": None,
    "status": "OK",
}


def test_parse_variant_extracts_all_fields():
    from ingestion.combo_loader import _parse_variant
    result = _parse_variant(FAKE_VARIANT)

    assert result["spellbook_id"] == "742-1295"
    assert result["color_identity"] == "B,G"
    assert result["popularity"] == 500
    assert result["card_names"] == ["Kitchen Finks", "Melira, Sylvok Outcast"]
    assert "Infinite life gain" in result["description"]


def test_parse_variant_handles_null_fields():
    from ingestion.combo_loader import _parse_variant
    result = _parse_variant(FAKE_VARIANT_NULLS)

    assert result["spellbook_id"] == "9-9"
    assert result["color_identity"] == ""
    assert result["popularity"] == 0
    assert result["card_names"] == []
    assert isinstance(result["description"], str)


def test_parse_variant_truncates_description_to_500_chars():
    from ingestion.combo_loader import _parse_variant
    long_variant = {**FAKE_VARIANT, "description": "X" * 600}
    result = _parse_variant(long_variant)
    assert len(result["description"]) <= 500


def test_infer_combo_type_infinite():
    from ingestion.combo_loader import _infer_combo_type
    assert _infer_combo_type("Infinite mana.") == "infinite"
    assert _infer_combo_type("Creates an INFINITE loop.") == "infinite"


def test_infer_combo_type_wincon():
    from ingestion.combo_loader import _infer_combo_type
    assert _infer_combo_type("Win the game.") == "wincon"
    assert _infer_combo_type("You win the game immediately.") == "wincon"


def test_infer_combo_type_synergy():
    from ingestion.combo_loader import _infer_combo_type
    assert _infer_combo_type("Draw a card.") == "synergy"
    assert _infer_combo_type("") == "synergy"


def _make_mock_driver():
    tx = MagicMock()
    tx.run.return_value.single.return_value = {"count": 2}
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.execute_write = MagicMock(
        side_effect=lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)
    )
    driver = MagicMock()
    driver.session.return_value = session
    return driver, tx, session


def test_load_combos_writes_combo_nodes():
    driver, tx, session = _make_mock_driver()

    with patch("ingestion.combo_loader.get_driver", return_value=driver):
        from ingestion.combo_loader import load_combos
        load_combos([FAKE_VARIANT])

    all_cypher = " ".join(str(c) for c in tx.run.call_args_list)
    assert "MERGE" in all_cypher
    assert "Combo" in all_cypher
    assert "spellbook_id" in all_cypher or "combo.spellbook_id" in all_cypher


def test_load_combos_writes_part_of_combo_relationships():
    driver, tx, session = _make_mock_driver()

    with patch("ingestion.combo_loader.get_driver", return_value=driver):
        from ingestion.combo_loader import load_combos
        load_combos([FAKE_VARIANT])

    all_cypher = " ".join(str(c) for c in tx.run.call_args_list)
    assert "PART_OF_COMBO" in all_cypher


def test_load_combos_returns_relationship_count():
    driver, tx, session = _make_mock_driver()

    with patch("ingestion.combo_loader.get_driver", return_value=driver):
        from ingestion.combo_loader import load_combos
        count = load_combos([FAKE_VARIANT])

    assert isinstance(count, int)
    assert count >= 0
