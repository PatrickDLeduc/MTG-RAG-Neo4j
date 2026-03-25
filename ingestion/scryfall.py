import httpx
import json
from pathlib import Path

BULK_DATA_URL = "https://api.scryfall.com/bulk-data"


async def fetch_bulk_data_url() -> str:
    """Get the download URL for the oracle_cards bulk data file."""
    async with httpx.AsyncClient() as client:
        response = await client.get(BULK_DATA_URL)
        response.raise_for_status()
        data = response.json()
        for item in data["data"]:
            if item["type"] == "oracle_cards":
                return item["download_uri"]
    raise ValueError("oracle_cards bulk data type not found in Scryfall response")


async def download_cards(cache_path: Path = Path("data/cards.json")) -> list[dict]:
    """Download oracle cards from Scryfall, using local cache if available."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = await fetch_bulk_data_url()

    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=180.0)
        response.raise_for_status()
        cards = response.json()

    cache_path.write_text(json.dumps(cards), encoding="utf-8")
    return cards
