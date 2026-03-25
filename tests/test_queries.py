from unittest.mock import MagicMock, patch


def _make_session(return_data):
    result = MagicMock()
    result.data.return_value = return_data
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = result
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


def test_vector_search_cards_returns_list():
    fake_cards = [{"id": "c1", "name": "A", "oracle_text": "text", "score": 0.9}]
    driver, session = _make_session(fake_cards)

    with patch("graph.queries.get_driver", return_value=driver):
        from graph.queries import vector_search_cards
        result = vector_search_cards([0.1] * 1536)

    assert result == fake_cards
    assert session.run.called


def test_expand_from_cards_returns_list():
    fake_expanded = [{"card_name": "A", "oracle_text": "txt", "keywords": ["Flying"], "combos": [], "combo_cards": []}]
    driver, session = _make_session(fake_expanded)

    with patch("graph.queries.get_driver", return_value=driver):
        from graph.queries import expand_from_cards
        result = expand_from_cards(["c1", "c2"])

    assert result == fake_expanded


def test_get_cards_by_keyword_returns_list():
    fake_cards = [{"id": "c1", "name": "Vampire A", "oracle_text": "..."}]
    driver, session = _make_session(fake_cards)

    with patch("graph.queries.get_driver", return_value=driver):
        from graph.queries import get_cards_by_keyword
        result = get_cards_by_keyword("Flying")

    assert result == fake_cards
    # Confirm the keyword was passed as a parameter
    call_kwargs = session.run.call_args
    assert "Flying" in str(call_kwargs)
