import asyncio
import httpx
import json
from pathlib import Path

VARIANTS_URL = "https://backend.commanderspellbook.com/variants/"
USER_AGENT = "MTG-RAG-Neo4j/1.0"


async def download_combos(
    cache_path: Path = Path("data/combos.json"),
) -> tuple[list[dict], bool]:
    """Download combo variants from Commander Spellbook, using local cache if available.

    Returns (combos, from_cache) so callers can report whether the API was hit.
    """
    if cache_path.exists():
        combos = json.loads(cache_path.read_text(encoding="utf-8"))
        return combos, True

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combos = await _fetch_all_variants()
    cache_path.write_text(json.dumps(combos), encoding="utf-8")
    return combos, False


async def _fetch_all_variants() -> list[dict]:
    results = []
    url = f"{VARIANTS_URL}?format=json&limit=100&ordering=popularity"

    async with httpx.AsyncClient(timeout=180.0, headers={"User-Agent": USER_AGENT}) as client:
        while url:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            page = [v for v in data["results"] if v.get("status") == "OK"]
            results.extend(page)
            url = data.get("next")
            if url:
                await asyncio.sleep(0.5)

    return results
