from unittest.mock import MagicMock, patch
import importlib


FAKE_VECTOR_RESULTS = [
    {"id": "c1", "name": "Kitchen Finks", "oracle_text": "When Kitchen Finks enters, you gain 2 life.", "score": 0.92},
]

FAKE_EXPANDED = [
    {
        "card_id": "c1",
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

    with patch("rag.retriever.vector_search_cards", return_value=FAKE_VECTOR_RESULTS), \
         patch("rag.retriever.search_cards_by_name", return_value=[]), \
         patch("rag.retriever.expand_from_cards", return_value=FAKE_EXPANDED):
        from rag import retriever
        retriever._client = mock_openai
        context = retriever.retrieve("What combos with Persist?")

    assert "Kitchen Finks" in context
    assert "Persist" in context


def test_retrieve_context_includes_combo_partners():
    mock_embedding = [0.1] * 1536
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value.data = [MagicMock(embedding=mock_embedding)]

    with patch("rag.retriever.vector_search_cards", return_value=FAKE_VECTOR_RESULTS), \
         patch("rag.retriever.search_cards_by_name", return_value=[]), \
         patch("rag.retriever.expand_from_cards", return_value=FAKE_EXPANDED):
        from rag import retriever
        retriever._client = mock_openai
        context = retriever.retrieve("What combos with Persist?")

    assert "Melira" in context


def test_detect_color_filter_colorless():
    from rag.retriever import _detect_color_filter
    assert _detect_color_filter("what colorless cards combo with X?") == "Colorless"


def test_detect_color_filter_blue():
    from rag.retriever import _detect_color_filter
    assert _detect_color_filter("best blue combo cards") == "Blue"


def test_detect_color_filter_none():
    from rag.retriever import _detect_color_filter
    assert _detect_color_filter("what combos with Prosper, Tome-Bound?") is None


def test_detect_color_filter_case_insensitive():
    from rag.retriever import _detect_color_filter
    assert _detect_color_filter("WHITE cards that gain life") == "White"


def test_retrieve_name_cards_appear_before_vector_cards():
    mock_embedding = [0.1] * 1536
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value.data = [MagicMock(embedding=mock_embedding)]

    fake_name_result = [{"id": "prosper", "name": "Prosper, Tome-Bound", "oracle_text": "..."}]
    fake_vector_result = [
        {"id": "prosperity", "name": "Prosperity", "oracle_text": "...", "score": 0.88},
        {"id": "tome", "name": "Tome of the Guildpact", "oracle_text": "...", "score": 0.85},
    ]
    fake_expanded = [
        {"card_id": "prosperity", "card_name": "Prosperity", "oracle_text": "...", "keywords": [], "combos": [], "combo_cards": []},
        {"card_id": "prosper", "card_name": "Prosper, Tome-Bound", "oracle_text": "...", "keywords": [], "combos": [], "combo_cards": []},
        {"card_id": "tome", "card_name": "Tome of the Guildpact", "oracle_text": "...", "keywords": [], "combos": [], "combo_cards": []},
    ]

    with patch("rag.retriever.search_cards_by_name", return_value=fake_name_result), \
         patch("rag.retriever.vector_search_cards", return_value=fake_vector_result), \
         patch("rag.retriever.expand_from_cards", return_value=fake_expanded):
        from rag import retriever
        retriever._client = mock_openai
        context = retriever.retrieve('what combos with "Prosper, Tome-Bound"?')

    assert context.index("Prosper, Tome-Bound") < context.index("Prosperity")


def test_retrieve_deduplicates_name_and_vector_results():
    mock_embedding = [0.1] * 1536
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value.data = [MagicMock(embedding=mock_embedding)]

    shared_card_vector = {"id": "prosper", "name": "Prosper, Tome-Bound", "oracle_text": "...", "score": 0.95}
    fake_expanded = [
        {"card_id": "prosper", "card_name": "Prosper, Tome-Bound", "oracle_text": "...",
         "keywords": [], "combos": [], "combo_cards": []},
    ]

    with patch("rag.retriever.search_cards_by_name", return_value=[{"id": "prosper", "name": "Prosper, Tome-Bound", "oracle_text": "..."}]), \
         patch("rag.retriever.vector_search_cards", return_value=[shared_card_vector]), \
         patch("rag.retriever.expand_from_cards") as mock_expand:
        mock_expand.return_value = fake_expanded
        from rag import retriever
        retriever._client = mock_openai
        retriever.retrieve('what combos with "Prosper, Tome-Bound"?')

    ids_passed = mock_expand.call_args[0][0]
    assert ids_passed.count("prosper") == 1


def test_retrieve_uses_color_filtered_search_when_color_detected():
    mock_embedding = [0.1] * 1536
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value.data = [MagicMock(embedding=mock_embedding)]

    with patch("rag.retriever.search_cards_by_name", return_value=[]), \
         patch("rag.retriever.vector_search_cards") as mock_unfiltered, \
         patch("rag.retriever.vector_search_cards_by_color", return_value=[]) as mock_filtered, \
         patch("rag.retriever.expand_from_cards", return_value=[]):
        from rag import retriever
        retriever._client = mock_openai
        retriever.retrieve("what colorless cards combo with X?")

    mock_filtered.assert_called_once()
    mock_unfiltered.assert_not_called()
    assert mock_filtered.call_args[0][1] == "Colorless"
