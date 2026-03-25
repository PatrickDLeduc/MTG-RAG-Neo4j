from unittest.mock import MagicMock, patch


def _fake_embedding(dim=1536):
    return [0.01] * dim


def test_embed_texts_calls_openai_and_returns_vectors():
    mock_item = MagicMock()
    mock_item.embedding = _fake_embedding()
    mock_response = MagicMock()
    mock_response.data = [mock_item]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client), \
         patch("config.OPENAI_API_KEY", "test_key"):
        from ingestion import embeddings as emb
        import importlib; importlib.reload(emb)  # force re-init with mock
        result = emb.embed_texts(["Flying is a keyword ability."])

    assert len(result) == 1
    assert len(result[0]) == 1536


def test_embed_cards_updates_all_unembedded_cards():
    mock_item = MagicMock()
    mock_item.embedding = _fake_embedding()
    mock_response = MagicMock()
    mock_response.data = [mock_item, mock_item]  # 2 cards in batch

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = mock_response

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value.data.return_value = [
        {"id": "c1", "text": "Oracle text one."},
        {"id": "c2", "text": "Oracle text two."},
    ]
    mock_driver = MagicMock()
    mock_driver.session.return_value = session

    with patch("config.OPENAI_API_KEY", "test_key"), \
         patch("openai.OpenAI", return_value=mock_openai):
        from ingestion import embeddings as emb
        import importlib; importlib.reload(emb)
        
        # Patch get_driver after reload
        with patch.object(emb, "get_driver", return_value=mock_driver):
            emb.embed_cards(batch_size=10)

    # Verify embeddings were written back
    assert session.execute_write.called
