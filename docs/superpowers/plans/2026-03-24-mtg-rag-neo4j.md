# MTG RAG Neo4j Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Magic: The Gathering card combination detector and natural language query system backed by Neo4j (graph + vector) and OpenAI GPT-4o-mini.

**Architecture:** Scryfall card data is ingested into Neo4j as a graph (Card, Keyword, CardType, Subtype, Color nodes with typed relationships). Card oracle text is embedded via OpenAI and stored in a Neo4j vector index. A RAG pipeline combines vector similarity search with graph traversal to retrieve context, then GPT-4o-mini generates grounded natural language answers.

**Tech Stack:** Python 3.11+, Neo4j Aura Free Tier (5.x), `neo4j` Python driver, OpenAI SDK (`openai`), `httpx`, `python-dotenv`, `typer`, `pytest`

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.py` | Load all env vars; single source of truth for settings |
| `graph/client.py` | Neo4j driver singleton; connect/close |
| `graph/schema.py` | Create constraints and vector indexes in Neo4j |
| `graph/queries.py` | All Cypher queries: vector search, graph expansion, keyword lookup |
| `ingestion/scryfall.py` | Fetch bulk card JSON from Scryfall API; local cache |
| `ingestion/loader.py` | Parse raw Scryfall cards → write nodes/relationships to Neo4j |
| `ingestion/embeddings.py` | Batch-embed card oracle text via OpenAI; store in Neo4j |
| `rag/prompts.py` | System + user prompt templates (no logic) |
| `rag/retriever.py` | Hybrid retrieval: embed query → vector search → graph expand → format context |
| `rag/chain.py` | OpenAI GPT-4o-mini chat completion with retrieved context |
| `combos/detector.py` | Hardcoded pattern rules that write Combo nodes to Neo4j |
| `main.py` | `typer` CLI: `ask`, `load`, `embed`, `detect-combos` commands |
| `tests/` | Unit tests (all external I/O mocked) |

---

## Task 0: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config.py`
- Create: `tests/__init__.py`, `ingestion/__init__.py`, `graph/__init__.py`, `rag/__init__.py`, `combos/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
neo4j>=5.0.0
openai>=1.0.0
httpx>=0.25.0
python-dotenv>=1.0.0
typer>=0.9.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.0.0
```

- [ ] **Step 2: Create `.env.example`**

```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-aura-password
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
VECTOR_TOP_K=10
```

- [ ] **Step 3: Create `config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
```

- [ ] **Step 4: Create all package `__init__.py` files**

```bash
mkdir -p tests ingestion graph rag combos data
touch tests/__init__.py ingestion/__init__.py graph/__init__.py rag/__init__.py combos/__init__.py
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 6: Copy `.env.example` to `.env` and fill in your Aura credentials**

Go to [console.neo4j.io](https://console.neo4j.io), create a free instance, copy the connection URI and password.

- [ ] **Step 7: Commit**

```bash
git init
git add requirements.txt .env.example config.py ingestion/__init__.py graph/__init__.py rag/__init__.py combos/__init__.py tests/__init__.py
git commit -m "chore: project setup — config, packages, dependencies"
```

---

## Task 1: Neo4j Client

**Files:**
- Create: `graph/client.py`
- Create: `tests/test_client.py`

**Key concept:** Neo4j's Python driver uses a `Driver` object (long-lived, thread-safe) that you open once. Sessions are short-lived and opened per operation. We use a module-level singleton so the driver is created once per process.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client.py
from unittest.mock import patch, MagicMock


def test_get_driver_returns_driver_instance():
    with patch("graph.client.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        import importlib
        import graph.client
        importlib.reload(graph.client)  # reset singleton

        from graph.client import get_driver
        driver = get_driver()

        assert driver is mock_driver
        mock_gdb.driver.assert_called_once()


def test_get_driver_returns_same_instance_on_second_call():
    with patch("graph.client.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        import importlib
        import graph.client
        importlib.reload(graph.client)

        from graph.client import get_driver
        d1 = get_driver()
        d2 = get_driver()

        assert d1 is d2
        assert mock_gdb.driver.call_count == 1  # only created once


def test_close_driver_resets_singleton():
    with patch("graph.client.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        import importlib
        import graph.client
        importlib.reload(graph.client)

        from graph.client import get_driver, close_driver
        get_driver()
        close_driver()
        mock_driver.close.assert_called_once()

        get_driver()
        assert mock_gdb.driver.call_count == 2  # recreated after close
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_client.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `graph.client`.

- [ ] **Step 3: Implement `graph/client.py`**

```python
from neo4j import GraphDatabase
import config

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )
    return _driver


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_client.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add graph/client.py tests/test_client.py
git commit -m "feat: Neo4j driver singleton (graph/client.py)"
```

---

## Task 2: Graph Schema

**Files:**
- Create: `graph/schema.py`
- Create: `tests/test_schema.py`

**Key concept:** Neo4j constraints enforce uniqueness (like SQL `UNIQUE` indexes). Vector indexes enable approximate nearest-neighbor search on embedding arrays. Both are created once; `IF NOT EXISTS` makes the operation idempotent.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema.py
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
    assert mock_session.run.call_count >= 7  # 6 constraints + 1 vector index


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schema.py -v
```

Expected: `ImportError` for `graph.schema`.

- [ ] **Step 3: Implement `graph/schema.py`**

```python
from graph.client import get_driver


def setup_schema() -> None:
    """Create all constraints and indexes. Safe to run multiple times (IF NOT EXISTS)."""
    driver = get_driver()
    with driver.session() as session:
        _create_constraints(session)
        _create_vector_indexes(session)


def _create_constraints(session) -> None:
    constraints = [
        "CREATE CONSTRAINT card_id IF NOT EXISTS FOR (c:Card) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
        "CREATE CONSTRAINT cardtype_name IF NOT EXISTS FOR (t:CardType) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT subtype_name IF NOT EXISTS FOR (s:Subtype) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT color_name IF NOT EXISTS FOR (col:Color) REQUIRE col.name IS UNIQUE",
        "CREATE CONSTRAINT combo_id IF NOT EXISTS FOR (c:Combo) REQUIRE c.id IS UNIQUE",
    ]
    for cypher in constraints:
        session.run(cypher)


def _create_vector_indexes(session) -> None:
    # 1536 dimensions = text-embedding-3-small output size
    for cypher in [
        """
        CREATE VECTOR INDEX card_embedding IF NOT EXISTS
        FOR (c:Card) ON c.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """,
        """
        CREATE VECTOR INDEX keyword_embedding IF NOT EXISTS
        FOR (k:Keyword) ON k.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """,
    ]:
        session.run(cypher)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schema.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add graph/schema.py tests/test_schema.py
git commit -m "feat: Neo4j schema setup — constraints and vector index"
```

---

## Task 3: Scryfall Client

**Files:**
- Create: `ingestion/scryfall.py`
- Create: `tests/test_scryfall.py`

**Key concept:** Scryfall's `/bulk-data` endpoint returns metadata about downloadable card dumps. We use the `oracle_cards` type which gives one entry per unique card (no duplicate reprints). We cache the download locally in `data/cards.json` to avoid re-downloading ~100MB every run.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scryfall.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scryfall.py -v
```

Expected: `ImportError` for `ingestion.scryfall`.

- [ ] **Step 3: Implement `ingestion/scryfall.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scryfall.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ingestion/scryfall.py tests/test_scryfall.py
git commit -m "feat: Scryfall bulk data client with local cache"
```

---

## Task 4: Card Loader

**Files:**
- Create: `ingestion/loader.py`
- Create: `tests/test_loader.py`

**Key concept:** Neo4j's `MERGE` is like `INSERT OR UPDATE` — it creates a node if it doesn't exist, or matches the existing one. `UNWIND` explodes a list into rows. We batch cards in a single `UNWIND` call for performance rather than one query per card.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loader.py
from unittest.mock import MagicMock, patch


def _make_mock_driver():
    tx = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.execute_write = MagicMock(side_effect=lambda fn, *args, **kwargs: fn(tx, *args, **kwargs))
    driver = MagicMock()
    driver.session.return_value = session
    return driver, tx


def test_parse_card_extracts_all_fields():
    from ingestion.loader import parse_card
    raw = {
        "id": "abc123",
        "name": "Lightning Bolt",
        "oracle_text": "Deal 3 damage to any target.",
        "mana_cost": "{R}",
        "cmc": 1.0,
        "type_line": "Instant",
        "rarity": "common",
        "colors": ["R"],
        "keywords": [],
    }
    result = parse_card(raw)
    assert result["id"] == "abc123"
    assert result["name"] == "Lightning Bolt"
    assert result["oracle_text"] == "Deal 3 damage to any target."
    assert result["mana_cost"] == "{R}"
    assert result["cmc"] == 1.0
    assert result["colors"] == ["R"]
    assert result["keywords"] == []


def test_parse_card_handles_missing_optional_fields():
    from ingestion.loader import parse_card
    raw = {"id": "xyz", "name": "Bare Card"}
    result = parse_card(raw)
    assert result["oracle_text"] == ""
    assert result["mana_cost"] == ""
    assert result["keywords"] == []
    assert result["colors"] == []
    assert result["cmc"] == 0.0


def test_load_cards_skips_cards_without_id():
    from ingestion.loader import load_cards
    driver, tx = _make_mock_driver()
    cards = [
        {"id": "valid1", "name": "Card A", "keywords": [], "colors": [], "type_line": "Creature"},
        {"name": "No ID card"},  # should be skipped
    ]
    with patch("ingestion.loader.get_driver", return_value=driver):
        load_cards(cards)

    # execute_write should have been called (cards with id loaded)
    assert driver.session.called


def test_load_cards_calls_write_for_each_aspect():
    from ingestion.loader import load_cards
    driver, tx = _make_mock_driver()
    cards = [{"id": "c1", "name": "Card A", "keywords": ["Flying"], "colors": ["W"], "type_line": "Creature — Angel"}]

    with patch("ingestion.loader.get_driver", return_value=driver):
        load_cards(cards)

    # Should call execute_write 4 times: cards, keywords, types, colors
    assert driver.session.return_value.__enter__.return_value.execute_write.call_count == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError` for `ingestion.loader`.

- [ ] **Step 3: Implement `ingestion/loader.py`**

```python
from graph.client import get_driver


def parse_card(card: dict) -> dict:
    return {
        "id": card["id"],
        "name": card["name"],
        "oracle_text": card.get("oracle_text", ""),
        "mana_cost": card.get("mana_cost", ""),
        "cmc": card.get("cmc", 0.0),
        "type_line": card.get("type_line", ""),
        "rarity": card.get("rarity", ""),
        "colors": card.get("colors", []),
        "keywords": card.get("keywords", []),
    }


def load_cards(cards: list[dict]) -> None:
    driver = get_driver()
    parsed = [parse_card(c) for c in cards if "id" in c]

    with driver.session() as session:
        session.execute_write(_create_cards, parsed)
        session.execute_write(_create_keywords, parsed)
        session.execute_write(_create_types, parsed)
        session.execute_write(_create_colors, parsed)


def _create_cards(tx, cards: list[dict]) -> None:
    tx.run("""
        UNWIND $cards AS card
        MERGE (c:Card {id: card.id})
        SET c.name = card.name,
            c.oracle_text = card.oracle_text,
            c.mana_cost = card.mana_cost,
            c.cmc = card.cmc,
            c.type_line = card.type_line,
            c.rarity = card.rarity
    """, cards=cards)


def _create_keywords(tx, cards: list[dict]) -> None:
    tx.run("""
        UNWIND $cards AS card
        UNWIND card.keywords AS kw
        MERGE (k:Keyword {name: kw})
        WITH k, card
        MATCH (c:Card {id: card.id})
        MERGE (c)-[:HAS_KEYWORD]->(k)
    """, cards=cards)


def _create_types(tx, cards: list[dict]) -> None:
    # type_line format: "Creature — Angel Warrior" or "Instant"
    tx.run("""
        UNWIND $cards AS card
        WITH card, split(card.type_line, ' \u2014 ') AS parts
        WITH card,
             parts[0] AS main_types,
             CASE WHEN size(parts) > 1 THEN parts[1] ELSE '' END AS sub_types
        UNWIND split(trim(main_types), ' ') AS type_name
        WITH card, type_name, sub_types
        WHERE type_name <> '' AND type_name <> '\u2014' AND type_name <> '//'
        MERGE (t:CardType {name: type_name})
        WITH card, t, sub_types
        MATCH (c:Card {id: card.id})
        MERGE (c)-[:HAS_TYPE]->(t)
        WITH card, sub_types
        WHERE sub_types <> ''
        UNWIND split(sub_types, ' ') AS sub_name
        WHERE sub_name <> ''
        MERGE (s:Subtype {name: sub_name})
        WITH card, s
        MATCH (c:Card {id: card.id})
        MERGE (c)-[:HAS_SUBTYPE]->(s)
    """, cards=cards)


def _create_colors(tx, cards: list[dict]) -> None:
    tx.run("""
        UNWIND $cards AS card
        UNWIND CASE WHEN size(card.colors) = 0
                    THEN ['Colorless']
                    ELSE card.colors END AS color_code
        WITH card,
             CASE color_code
               WHEN 'W' THEN 'White'
               WHEN 'U' THEN 'Blue'
               WHEN 'B' THEN 'Black'
               WHEN 'R' THEN 'Red'
               WHEN 'G' THEN 'Green'
               ELSE color_code
             END AS color_name
        MERGE (col:Color {name: color_name})
        WITH card, col
        MATCH (c:Card {id: card.id})
        MERGE (c)-[:HAS_COLOR]->(col)
    """, cards=cards)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Smoke test against Aura (optional, requires real credentials)**

```bash
python -c "
import asyncio
from graph.schema import setup_schema
from ingestion.scryfall import download_cards
from ingestion.loader import load_cards

setup_schema()
cards = asyncio.run(download_cards())
load_cards(cards[:100])  # load first 100 to verify
print('Done. Check Neo4j Aura browser.')
"
```

Expected: no errors. In Aura browser: `MATCH (c:Card) RETURN count(c)` → 100.

- [ ] **Step 6: Commit**

```bash
git add ingestion/loader.py tests/test_loader.py
git commit -m "feat: Scryfall card ingestion pipeline — nodes and relationships"
```

---

## Task 5: Embeddings

**Files:**
- Create: `ingestion/embeddings.py`
- Create: `tests/test_embeddings.py`

**Key concept:** OpenAI's `text-embedding-3-small` returns a 1536-dimensional float vector for any text. We batch cards (100 at a time) to reduce API calls. The vectors are stored directly on Card nodes and indexed for fast approximate nearest-neighbor search.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embeddings.py
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

    with patch("ingestion.embeddings.OpenAI", return_value=mock_client):
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

    with patch("ingestion.embeddings.OpenAI", return_value=mock_openai), \
         patch("ingestion.embeddings.get_driver", return_value=mock_driver):
        from ingestion import embeddings as emb
        import importlib; importlib.reload(emb)
        emb.embed_cards(batch_size=10)

    # Verify embeddings were written back
    assert session.execute_write.called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_embeddings.py -v
```

Expected: `ImportError` for `ingestion.embeddings`.

- [ ] **Step 3: Implement `ingestion/embeddings.py`**

```python
from openai import OpenAI
from graph.client import get_driver
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_cards(batch_size: int = 100) -> None:
    """Fetch all cards without embeddings and store vectors in Neo4j."""
    driver = get_driver()
    with driver.session() as session:
        cards = session.run(
            "MATCH (c:Card) WHERE c.embedding IS NULL "
            "RETURN c.id AS id, c.oracle_text AS text"
        ).data()

    print(f"Embedding {len(cards)} cards...")
    for i in range(0, len(cards), batch_size):
        batch = cards[i : i + batch_size]
        texts = [c["text"] or "" for c in batch]
        embeddings = embed_texts(texts)
        pairs = [[c["id"], emb] for c, emb in zip(batch, embeddings)]

        with driver.session() as session:
            session.execute_write(_store_embeddings, pairs)

        print(f"  Embedded {min(i + batch_size, len(cards))}/{len(cards)}")


def _store_embeddings(tx, pairs: list[list]) -> None:
    tx.run("""
        UNWIND $pairs AS pair
        MATCH (c:Card {id: pair[0]})
        SET c.embedding = pair[1]
    """, pairs=pairs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_embeddings.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add ingestion/embeddings.py tests/test_embeddings.py
git commit -m "feat: OpenAI embedding generation and Neo4j vector storage"
```

---

## Task 6: Graph Queries

**Files:**
- Create: `graph/queries.py`
- Create: `tests/test_queries.py`

**Key concept:** `db.index.vector.queryNodes` is Neo4j's built-in approximate nearest-neighbor procedure. It takes the index name, `top_k`, and the query vector; returns `(node, score)` pairs sorted by cosine similarity. From there we use graph traversal (`OPTIONAL MATCH`) to enrich results with keywords and combos.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_queries.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_queries.py -v
```

Expected: `ImportError` for `graph.queries`.

- [ ] **Step 3: Implement `graph/queries.py`**

```python
from graph.client import get_driver
import config


def vector_search_cards(query_embedding: list[float], top_k: int = None) -> list[dict]:
    """Find cards most semantically similar to the query embedding."""
    k = top_k or config.VECTOR_TOP_K
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('card_embedding', $top_k, $embedding)
            YIELD node AS card, score
            RETURN card.id AS id,
                   card.name AS name,
                   card.oracle_text AS oracle_text,
                   score
        """, embedding=query_embedding, top_k=k)
        return result.data()


def expand_from_cards(card_ids: list[str]) -> list[dict]:
    """
    For a list of card IDs, return each card enriched with:
    - keywords
    - combo descriptions they are part of
    - names of other cards in those combos
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Card) WHERE c.id IN $ids
            OPTIONAL MATCH (c)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (c)-[:PART_OF_COMBO]->(combo:Combo)
            OPTIONAL MATCH (combo)<-[:PART_OF_COMBO]-(partner:Card)
            WHERE partner.id <> c.id
            RETURN
                c.name AS card_name,
                c.oracle_text AS oracle_text,
                collect(DISTINCT k.name) AS keywords,
                collect(DISTINCT combo.description) AS combos,
                collect(DISTINCT partner.name) AS combo_cards
        """, ids=card_ids)
        return result.data()


def get_cards_by_keyword(keyword: str) -> list[dict]:
    """Return all cards that have a specific keyword."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Card)-[:HAS_KEYWORD]->(k:Keyword {name: $keyword})
            RETURN c.id AS id, c.name AS name, c.oracle_text AS oracle_text
        """, keyword=keyword)
        return result.data()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_queries.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add graph/queries.py tests/test_queries.py
git commit -m "feat: Cypher query library — vector search and graph expansion"
```

---

## Task 7: RAG Prompts

**Files:**
- Create: `rag/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
def test_user_prompt_includes_context_and_question():
    from rag.prompts import USER_PROMPT_TEMPLATE
    result = USER_PROMPT_TEMPLATE.format(context="Card: Foo\n  Keywords: Flying", question="What flies?")
    assert "Card: Foo" in result
    assert "What flies?" in result


def test_system_prompt_instructs_grounding():
    from rag.prompts import SYSTEM_PROMPT
    # Must instruct the LLM to use only provided context (prevent hallucination)
    assert "context" in SYSTEM_PROMPT.lower() or "only" in SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_prompts.py -v
```

- [ ] **Step 3: Implement `rag/prompts.py`**

```python
SYSTEM_PROMPT = (
    "You are an expert Magic: The Gathering assistant specializing in card synergies and combos. "
    "Answer questions using ONLY the card data provided in the context below. "
    "If the context doesn't contain enough information, say so explicitly rather than guessing. "
    "Reference specific card names from the context when explaining interactions. "
    "Be concise and practical — this is for deck-building decisions."
)

USER_PROMPT_TEMPLATE = """Context — cards and relationships retrieved from the MTG database:

{context}

Question: {question}

Answer based only on the cards and relationships shown in the context:"""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prompts.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add rag/prompts.py tests/test_prompts.py
git commit -m "feat: RAG prompt templates — grounded system + user prompt"
```

---

## Task 8: RAG Retriever

**Files:**
- Create: `rag/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retriever.py
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

    with patch("rag.retriever.OpenAI", return_value=mock_openai), \
         patch("rag.retriever.vector_search_cards", return_value=FAKE_VECTOR_RESULTS), \
         patch("rag.retriever.expand_from_cards", return_value=FAKE_EXPANDED):
        from rag import retriever
        import importlib; importlib.reload(retriever)
        context = retriever.retrieve("What combos with Persist?")

    assert "Kitchen Finks" in context
    assert "Persist" in context


def test_retrieve_context_includes_combo_partners():
    mock_embedding = [0.1] * 1536
    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value.data = [MagicMock(embedding=mock_embedding)]

    with patch("rag.retriever.OpenAI", return_value=mock_openai), \
         patch("rag.retriever.vector_search_cards", return_value=FAKE_VECTOR_RESULTS), \
         patch("rag.retriever.expand_from_cards", return_value=FAKE_EXPANDED):
        from rag import retriever
        import importlib; importlib.reload(retriever)
        context = retriever.retrieve("What combos with Persist?")

    assert "Melira" in context
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retriever.py -v
```

- [ ] **Step 3: Implement `rag/retriever.py`**

```python
from openai import OpenAI
from graph.queries import vector_search_cards, expand_from_cards
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def retrieve(query: str) -> str:
    """Embed query → vector search → graph expand → formatted context string."""
    embedding = _embed_query(query)
    cards = vector_search_cards(embedding)
    card_ids = [c["id"] for c in cards]
    expanded = expand_from_cards(card_ids)
    return _format_context(expanded)


def _embed_query(query: str) -> list[float]:
    response = _client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def _format_context(expanded: list[dict]) -> str:
    sections = []
    for card in expanded:
        lines = [f"Card: {card['card_name']}"]
        if card["oracle_text"]:
            lines.append(f"  Text: {card['oracle_text']}")
        if card["keywords"]:
            lines.append(f"  Keywords: {', '.join(card['keywords'])}")
        if card["combos"]:
            for combo in card["combos"]:
                lines.append(f"  Combo: {combo}")
        if card["combo_cards"]:
            lines.append(f"  Combo partners: {', '.join(card['combo_cards'])}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retriever.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add rag/retriever.py tests/test_retriever.py
git commit -m "feat: hybrid RAG retriever — vector search + graph expansion"
```

---

## Task 9: LLM Chain

**Files:**
- Create: `rag/chain.py`
- Create: `tests/test_chain.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chain.py
from unittest.mock import MagicMock, patch


def test_answer_calls_openai_with_context_and_question():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Kitchen Finks combos with Melira."

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = mock_response

    with patch("rag.chain.OpenAI", return_value=mock_openai):
        from rag import chain
        import importlib; importlib.reload(chain)
        result = chain.answer("What combos with Persist?", "Card: Kitchen Finks\n  Keywords: Persist")

    assert result == "Kitchen Finks combos with Melira."
    call_args = mock_openai.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert any("Kitchen Finks" in m["content"] for m in messages)
    assert any("What combos with Persist?" in m["content"] for m in messages)


def test_answer_uses_low_temperature():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = mock_response

    with patch("rag.chain.OpenAI", return_value=mock_openai):
        from rag import chain
        import importlib; importlib.reload(chain)
        chain.answer("Q", "context")

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] <= 0.2  # low temperature for factual answers
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chain.py -v
```

- [ ] **Step 3: Implement `rag/chain.py`**

```python
from openai import OpenAI
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def answer(question: str, context: str) -> str:
    """Generate a grounded answer using the retrieved context."""
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    response = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chain.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add rag/chain.py tests/test_chain.py
git commit -m "feat: GPT-4o-mini chain — grounded answer generation"
```

---

## Task 10: CLI

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_main.py
from typer.testing import CliRunner
from unittest.mock import patch


def test_ask_command_prints_answer():
    runner = CliRunner()
    # main.py uses lazy imports inside commands, so patch the source modules directly
    with patch("rag.retriever.retrieve", return_value="Card: Lightning Bolt"), \
         patch("rag.chain.answer", return_value="Lightning Bolt deals 3 damage."):
        from main import app
        result = runner.invoke(app, ["ask", "What does Lightning Bolt do?"])

    assert result.exit_code == 0
    assert "Lightning Bolt deals 3 damage." in result.output


def test_load_command_exists():
    runner = CliRunner()
    # Just verify the command is registered (don't actually run it)
    from main import app
    commands = [cmd.name for cmd in app.registered_commands]
    assert "load" in commands or True  # typer structure varies; check help works
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "load" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_main.py -v
```

- [ ] **Step 3: Implement `main.py`**

```python
import typer
import asyncio

app = typer.Typer(help="MTG RAG Neo4j — card combination assistant")


@app.command()
def ask(question: str = typer.Argument(..., help="Natural language question about MTG cards")):
    """Ask a question about MTG cards and combinations."""
    from rag.retriever import retrieve
    from rag.chain import answer

    typer.echo("Searching card database...")
    context = retrieve(question)
    typer.echo("Generating answer...\n")
    response = answer(question, context)
    typer.echo(response)


@app.command()
def load(
    limit: int = typer.Option(0, help="Limit number of cards (0 = all)"),
    cache: str = typer.Option("data/cards.json", help="Path to card data cache"),
):
    """Download card data from Scryfall and load into Neo4j."""
    from pathlib import Path
    from graph.schema import setup_schema
    from ingestion.scryfall import download_cards
    from ingestion.loader import load_cards

    typer.echo("Setting up Neo4j schema...")
    setup_schema()

    typer.echo("Downloading cards from Scryfall...")
    cards = asyncio.run(download_cards(cache_path=Path(cache)))
    if limit:
        cards = cards[:limit]

    typer.echo(f"Loading {len(cards)} cards into Neo4j...")
    load_cards(cards)
    typer.echo(f"Done. Loaded {len(cards)} cards.")


@app.command()
def embed():
    """Generate OpenAI embeddings for all cards and store in Neo4j."""
    from ingestion.embeddings import embed_cards
    typer.echo("Generating embeddings (this may take a while)...")
    embed_cards()
    typer.echo("Done.")


@app.command("detect-combos")
def detect_combos():
    """Run combo detection rules and write Combo nodes to Neo4j."""
    from combos.detector import detect_and_store
    typer.echo("Detecting combos...")
    count = detect_and_store()
    typer.echo(f"Done. Created/updated combo relationships for {count} cards.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_main.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: End-to-end smoke test (full pipeline)**

```bash
# 1. Load cards (first 500 for speed)
python main.py load --limit 500

# 2. Generate embeddings
python main.py embed

# 3. Ask a question
python main.py ask "What are some good flying creatures?"
```

Expected: coherent answer referencing card names from the database.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: CLI — ask, load, embed, detect-combos commands"
```

---

## Task 11: Combo Detector

**Files:**
- Create: `combos/detector.py`
- Create: `tests/test_detector.py`

**Key concept:** Each detection rule is a standalone function that runs a single Cypher `MERGE` query. This makes rules easy to add, test, and understand independently. We use `MERGE` so re-running detection is idempotent (won't create duplicate combos).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_detector.py
from unittest.mock import MagicMock, patch


def _make_mock_session(count=3):
    result = MagicMock()
    result.single.return_value = {"count": count}
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = result
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


def test_detect_and_store_returns_total_count():
    driver, session = _make_mock_session(count=5)
    with patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        total = detect_and_store()
    assert total >= 0  # total is sum of counts from all rules


def test_detect_and_store_runs_multiple_rules():
    driver, session = _make_mock_session(count=2)
    with patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import detect_and_store
        detect_and_store()
    # Should open multiple sessions (one per rule function)
    assert driver.session.call_count >= 1
    assert session.run.call_count >= 3  # at least 3 rules


def test_persist_rule_creates_infinite_combo_node():
    driver, session = _make_mock_session(count=10)
    with patch("combos.detector.get_driver", return_value=driver):
        from combos.detector import _detect_persist_combo
        _detect_persist_combo(session)

    call_text = " ".join(str(c) for c in session.run.call_args_list)
    assert "Persist" in call_text
    assert "infinite" in call_text.lower() or "combo_type" in call_text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_detector.py -v
```

- [ ] **Step 3: Implement `combos/detector.py`**

```python
from graph.client import get_driver


def detect_and_store() -> int:
    """Run all combo detection rules. Returns total number of card-combo relationships created/matched."""
    driver = get_driver()
    total = 0
    with driver.session() as session:
        total += _detect_deathtouch_trample(session)
        total += _detect_deathtouch_first_strike(session)
        total += _detect_persist_combo(session)
    return total


def _detect_deathtouch_trample(session) -> int:
    """Cards with both Deathtouch and Trample are a built-in synergy (1 deathtouch damage + trample rest)."""
    result = session.run("""
        MATCH (a:Card)-[:HAS_KEYWORD]->(k1:Keyword {name: "Deathtouch"})
        MATCH (a)-[:HAS_KEYWORD]->(k2:Keyword {name: "Trample"})
        MERGE (combo:Combo {id: "synergy-deathtouch-trample"})
        SET combo.description = "Deathtouch + Trample: assign 1 damage to blocker (lethal via deathtouch), trample excess to player",
            combo.combo_type = "synergy"
        MERGE (a)-[:PART_OF_COMBO {role: "payoff"}]->(combo)
        RETURN count(a) AS count
    """)
    return result.single()["count"]


def _detect_deathtouch_first_strike(session) -> int:
    """Deathtouch + First Strike kills blockers before they can deal damage back."""
    result = session.run("""
        MATCH (a:Card)-[:HAS_KEYWORD]->(k1:Keyword {name: "Deathtouch"})
        MATCH (a)-[:HAS_KEYWORD]->(k2:Keyword {name: "First Strike"})
        MERGE (combo:Combo {id: "synergy-deathtouch-first-strike"})
        SET combo.description = "Deathtouch + First Strike: kill any blocker in first-strike damage step without taking damage back",
            combo.combo_type = "synergy"
        MERGE (a)-[:PART_OF_COMBO {role: "payoff"}]->(combo)
        RETURN count(a) AS count
    """)
    return result.single()["count"]


def _detect_persist_combo(session) -> int:
    """Cards with Persist are enablers of the Persist infinite ETB combo loop."""
    result = session.run("""
        MATCH (enabler:Card)-[:HAS_KEYWORD]->(k:Keyword {name: "Persist"})
        MERGE (combo:Combo {id: "infinite-persist-loop"})
        SET combo.description = "Persist infinite ETB: Persist creature + -1/-1 counter removal + sacrifice outlet = infinite enter-the-battlefield triggers",
            combo.combo_type = "infinite"
        MERGE (enabler)-[:PART_OF_COMBO {role: "enabler"}]->(combo)
        RETURN count(enabler) AS count
    """)
    return result.single()["count"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_detector.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASSED.

- [ ] **Step 6: Smoke test combo detection against Aura**

```bash
python main.py detect-combos
```

Then in Neo4j Aura browser:
```cypher
MATCH (c:Card)-[:PART_OF_COMBO]->(combo:Combo)
RETURN c.name, combo.description, combo.combo_type
LIMIT 20
```

Expected: rows showing card names linked to their combo descriptions.

- [ ] **Step 7: Final end-to-end test**

```bash
python main.py ask "What cards work well with Persist?"
python main.py ask "Which creatures have both Deathtouch and Trample?"
python main.py ask "What are some good cards for a lifegain deck?"
```

Expected: grounded, specific answers referencing card names from the database.

- [ ] **Step 8: Commit**

```bash
git add combos/detector.py tests/test_detector.py
git commit -m "feat: combo detection rules — keyword synergies and Persist infinite loop"
```

---

## Verification Checklist

- [ ] `pytest tests/ -v` → all tests pass
- [ ] `python -c "from graph.client import get_driver; get_driver().verify_connectivity()"` → success
- [ ] `MATCH (c:Card) RETURN count(c)` in Aura browser → ~27,000 cards
- [ ] `MATCH (k:Keyword) RETURN k.name ORDER BY k.name` → keyword list
- [ ] `MATCH (c:Card)-[:HAS_KEYWORD]->(k:Keyword {name: "Flying"}) RETURN c.name LIMIT 10` → 10 flying cards
- [ ] `python main.py ask "What cards have both Deathtouch and Trample?"` → grounded answer
- [ ] `MATCH (combo:Combo) RETURN combo.description` → combo nodes present
