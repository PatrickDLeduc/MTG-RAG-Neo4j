from unittest.mock import MagicMock, patch, call


def _make_mock_session():
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


def test_setup_schema_creates_constraints():
    mock_session = _make_mock_session()
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("graph.schema.get_driver", return_value=mock_driver):
        from graph.schema import setup_schema
        setup_schema()

    # Verify session.run was called multiple times (constraints + indexes)
    assert mock_session.run.call_count >= 7  # 6 constraints + at least 1 vector index


def test_setup_schema_includes_card_constraint():
    mock_session = _make_mock_session()
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("graph.schema.get_driver", return_value=mock_driver):
        from graph.schema import setup_schema
        setup_schema()

    calls_text = " ".join(str(c) for c in mock_session.run.call_args_list)
    assert "Card" in calls_text
    assert "UNIQUE" in calls_text


def test_setup_schema_creates_vector_index():
    mock_session = _make_mock_session()
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("graph.schema.get_driver", return_value=mock_driver):
        from graph.schema import setup_schema
        setup_schema()

    calls_text = " ".join(str(c) for c in mock_session.run.call_args_list)
    assert "VECTOR INDEX" in calls_text.upper() or "vector" in calls_text
