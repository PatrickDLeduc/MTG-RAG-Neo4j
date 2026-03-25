import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


FAKE_BULK_RESPONSE = {
    "data": [
        {"type": "all_cards", "download_uri": "https://scryfall.com/all.json"},
        {"type": "oracle_cards", "download_uri": "https://scryfall.com/oracle.json"},
    ]
}

FAKE_CARDS = [
    {"id": "abc", "name": "Lightning Bolt", "oracle_text": "Deal 3 damage.", "keywords": []},
]


@pytest.mark.asyncio
async def test_fetch_bulk_data_url_returns_oracle_uri():
    mock_response = MagicMock()
    mock_response.json.return_value = FAKE_BULK_RESPONSE
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("ingestion.scryfall.httpx.AsyncClient", return_value=mock_client):
        from ingestion.scryfall import fetch_bulk_data_url
        url = await fetch_bulk_data_url()

    assert url == "https://scryfall.com/oracle.json"


@pytest.mark.asyncio
async def test_download_cards_uses_cache_when_present(tmp_path):
    cache_file = tmp_path / "cards.json"
    cache_file.write_text(json.dumps(FAKE_CARDS))

    from ingestion.scryfall import download_cards
    cards = await download_cards(cache_path=cache_file)

    assert len(cards) == 1
    assert cards[0]["name"] == "Lightning Bolt"


@pytest.mark.asyncio
async def test_download_cards_writes_cache_on_first_run(tmp_path):
    cache_file = tmp_path / "cards.json"

    mock_bulk_response = MagicMock()
    mock_bulk_response.json.return_value = FAKE_BULK_RESPONSE
    mock_bulk_response.raise_for_status = MagicMock()

    mock_cards_response = MagicMock()
    mock_cards_response.json.return_value = FAKE_CARDS
    mock_cards_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_bulk_response, mock_cards_response]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("ingestion.scryfall.httpx.AsyncClient", return_value=mock_client):
        from ingestion.scryfall import download_cards
        cards = await download_cards(cache_path=cache_file)

    assert cache_file.exists()
    assert cards[0]["name"] == "Lightning Bolt"
