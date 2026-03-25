import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


FAKE_PAGE_1 = {
    "count": None,
    "next": "https://backend.commanderspellbook.com/variants/?format=json&limit=1&offset=1",
    "previous": None,
    "results": [
        {"id": "1-2", "status": "OK", "uses": [], "identity": "W",
         "popularity": 100, "description": "Infinite mana loop."},
    ],
}
FAKE_PAGE_2 = {
    "count": None,
    "next": None,
    "previous": None,
    "results": [
        {"id": "3-4", "status": "DRAFT", "uses": [], "identity": "U", "popularity": 5,
         "description": "Draft combo."},
    ],
}


@pytest.mark.asyncio
async def test_download_combos_uses_cache_when_present(tmp_path):
    cache_file = tmp_path / "combos.json"
    cached = [{"id": "1-2", "status": "OK"}]
    cache_file.write_text(json.dumps(cached))

    from ingestion.spellbook import download_combos
    result = await download_combos(cache_path=cache_file)

    assert result == cached


@pytest.mark.asyncio
async def test_download_combos_writes_cache_on_first_run(tmp_path):
    cache_file = tmp_path / "combos.json"

    mock_response_1 = MagicMock()
    mock_response_1.json.return_value = FAKE_PAGE_1
    mock_response_1.raise_for_status = MagicMock()

    mock_response_2 = MagicMock()
    mock_response_2.json.return_value = FAKE_PAGE_2
    mock_response_2.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_response_1, mock_response_2]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("ingestion.spellbook.httpx.AsyncClient", return_value=mock_client), \
         patch("ingestion.spellbook.asyncio.sleep", new=AsyncMock()):
        from ingestion.spellbook import download_combos
        result = await download_combos(cache_path=cache_file)

    assert cache_file.exists()
    # Only status==OK variant from page 1 is kept; DRAFT from page 2 is filtered
    assert len(result) == 1
    assert result[0]["id"] == "1-2"


@pytest.mark.asyncio
async def test_download_combos_filters_non_ok_status(tmp_path):
    cache_file = tmp_path / "combos.json"

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "count": None, "next": None, "previous": None,
        "results": [
            {"id": "5-6", "status": "DRAFT", "uses": []},
            {"id": "7-8", "status": "OK", "uses": []},
        ],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("ingestion.spellbook.httpx.AsyncClient", return_value=mock_client), \
         patch("ingestion.spellbook.asyncio.sleep", new=AsyncMock()):
        from ingestion.spellbook import download_combos
        result = await download_combos(cache_path=cache_file)

    assert len(result) == 1
    assert result[0]["id"] == "7-8"


@pytest.mark.asyncio
async def test_download_combos_paginates_until_next_is_null(tmp_path):
    cache_file = tmp_path / "combos.json"

    responses = [
        MagicMock(**{"json.return_value": FAKE_PAGE_1, "raise_for_status": MagicMock()}),
        MagicMock(**{"json.return_value": FAKE_PAGE_2, "raise_for_status": MagicMock()}),
    ]

    mock_client = AsyncMock()
    mock_client.get.side_effect = responses
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("ingestion.spellbook.httpx.AsyncClient", return_value=mock_client), \
         patch("ingestion.spellbook.asyncio.sleep", new=AsyncMock()):
        from ingestion.spellbook import download_combos
        await download_combos(cache_path=cache_file)

    assert mock_client.get.call_count == 2
