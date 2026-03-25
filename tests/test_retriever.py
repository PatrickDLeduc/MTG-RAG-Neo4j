from unittest.mock import MagicMock, patch


FAKE_VECTOR_RESULTS = [
    {"id": "c1", "name": "Kitchen Finks", "oracle_text": "When Kitchen Finks enters, you gain 2 life.", "score": 0.92},
]

FAKE_EXPANDED = [
    {
        "card_name": "Kitchen Finks",
        "oracle_text": "When Kitchen Finks enters, you gain 2 life.",
        "keywords": ["Persist"],
        "combos": ["Persist combo: infinite ETB"],
        "combo_cards": ["Melira, Sylvok Outcast"],
    }
]


def test_retrieve_returns_formatted_string():
    mock_embedding = [0.1] * 1536
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value.data = [MagicMock(embedding=mock_embedding)]

    with patch("rag.retriever._get_client", return_value=mock_openai), \
         patch("rag.retriever.vector_search_cards", return_value=FAKE_VECTOR_RESULTS), \
         patch("rag.retriever.expand_from_cards", return_value=FAKE_EXPANDED):
        from rag import retriever
        context = retriever.retrieve("What combos with Persist?")

    assert "Kitchen Finks" in context
    assert "Persist" in context


def test_retrieve_context_includes_combo_partners():
    mock_embedding = [0.1] * 1536
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value.data = [MagicMock(embedding=mock_embedding)]

    with patch("rag.retriever._get_client", return_value=mock_openai), \
         patch("rag.retriever.vector_search_cards", return_value=FAKE_VECTOR_RESULTS), \
         patch("rag.retriever.expand_from_cards", return_value=FAKE_EXPANDED):
        from rag import retriever
        context = retriever.retrieve("What combos with Persist?")

    assert "Melira" in context
