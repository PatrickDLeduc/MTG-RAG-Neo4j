# MTG RAG Neo4j

A Magic: The Gathering card combination assistant powered by hybrid RAG — combining vector search with Neo4j graph traversal to answer natural language questions about card synergies.

## Overview

Ask questions like *"What cards synergize with deathtouch?"* or *"Which cards work well with persist?"* and get grounded answers backed by real card data. The system embeds your query, finds semantically similar cards via vector search, expands the results through the card relationship graph (keywords, combos, types), and feeds the context to GPT-4o-mini to generate a focused answer.

Pure vector search misses structural relationships; pure graph traversal misses semantic similarity. The hybrid approach captures both.

## How It Works

```
INGESTION PIPELINE (run once)

  Scryfall API                  Commander Spellbook API
      |                                  |
      v                                  v
  ingestion/scryfall.py          ingestion/spellbook.py
  ──>  data/cards.json (cache)   ──>  data/combos.json (cache)
      |                                  |
      v                                  v
  ingestion/loader.py            ingestion/combo_loader.py
  ──>  Neo4j nodes:              ──>  Combo nodes (68,000+)
       Card, Keyword,                 + PART_OF_COMBO relationships
       CardType, Subtype, Color
      |
      v
  ingestion/embeddings.py ─>  OpenAI text-embedding-3-small
                              ──>  Card.embedding stored in Neo4j (1536-dim, cosine)


QUERY PIPELINE

  User question
      |
      v
  rag/retriever.py  ──>  embed question  ──>  vector search (top-10 cards)
                                              ──>  graph expand (keywords, combos)
                                              ──>  format context string
      |
      v
  rag/chain.py  ──>  GPT-4o-mini  ──>  printed answer
```

## Tech Stack

| Component        | Technology                                  |
|------------------|---------------------------------------------|
| Language         | Python 3.11+                                |
| Graph database   | Neo4j 5.x (Aura Free Tier)                  |
| Embeddings       | OpenAI `text-embedding-3-small` (1536-dim)  |
| LLM              | OpenAI `gpt-4o-mini`                        |
| CLI              | Typer 0.9+                                  |
| HTTP client      | httpx 0.25+                                 |
| Testing          | pytest + pytest-asyncio + pytest-mock       |

## Prerequisites

- Python 3.11+
- A [Neo4j Aura](https://console.neo4j.io) free instance (connection URI + password)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
git clone <repo-url>
cd "MTG RAG Neo4j"
pip install -r requirements.txt
```

Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
OPENAI_API_KEY=sk-...
```

## Setup

Run these three commands once to populate the database:

```bash
# 1. Fetch all cards from Scryfall and build the graph
python -m main load

# 2. Generate OpenAI embeddings for every card and store them in Neo4j
python -m main embed

# 3. Fetch 68,000+ combos from Commander Spellbook and write Combo nodes
python main.py detect-combos               # loads all combos (~68k)
python main.py detect-combos --limit 10000 # top 10k by popularity (safer for Neo4j free tier)
```

The `load` command caches the Scryfall bulk data to `data/cards.json`. The `detect-combos` command caches combo data to `data/combos.json`. Both skip the download on subsequent runs.

## Usage

### ask

Ask a natural language question about cards and combinations.

```bash
python -m main ask "What cards synergize with deathtouch?"
python -m main ask "Which green creatures have persist?"
python -m main ask "What combos exist with first strike?"
```

### load

Download card data from Scryfall and load it into Neo4j.

```bash
python -m main load                          # load all cards (default)
python -m main load --limit 100              # load only 100 cards (for testing)
python -m main load --cache data/cards.json  # specify cache file path
```

Options:
- `--limit` — number of cards to load; `0` means all (default: `0`)
- `--cache` — path to local JSON cache (default: `data/cards.json`)

### embed

Generate OpenAI embeddings for all cards and store them in Neo4j.

```bash
python -m main embed
```

Only processes cards that don't already have embeddings. Safe to re-run.

### detect-combos

Fetch combo data from [Commander Spellbook](https://commanderspellbook.com/) and write 68,000+ `Combo` nodes to Neo4j.

```bash
python main.py detect-combos                           # loads all ~68k combos
python main.py detect-combos --limit 10000             # top 10k by popularity
python main.py detect-combos --cache data/combos.json  # specify cache path
```

Combos are sorted by EDHREC deck count (most popular first), so `--limit` gives you the most widely-played combos. The data is cached locally after the first fetch.

Options:
- `--limit` — number of combos to load; `0` means all (default: `0`)
- `--cache` — path to local JSON cache (default: `data/combos.json`)

> **Neo4j Aura Free Tier note:** The free tier caps at ~200k nodes and ~400k relationships. If you already have cards loaded, use `--limit 10000` to stay within limits.

## Graph Schema

### Nodes

| Label      | Unique property | Key properties                                          |
|------------|----------------|---------------------------------------------------------|
| `Card`     | `id`           | `name`, `oracle_text`, `mana_cost`, `cmc`, `type_line`, `rarity`, `embedding` |
| `Keyword`  | `name`         | `embedding`                                             |
| `CardType` | `name`         |                                                         |
| `Subtype`  | `name`         |                                                         |
| `Color`    | `name`         |                                                         |
| `Combo`    | `id`           | `description`, `combo_type`, `color_identity`, `popularity`, `source` |

### Relationships

| Relationship             | From   | To         | Properties       |
|--------------------------|--------|------------|------------------|
| `HAS_KEYWORD`            | Card   | Keyword    |                  |
| `HAS_TYPE`               | Card   | CardType   |                  |
| `HAS_SUBTYPE`            | Card   | Subtype    |                  |
| `HAS_COLOR`              | Card   | Color      |                  |
| `PART_OF_COMBO`          | Card   | Combo      | `role` (payoff / enabler) |

### Indexes

- Vector index on `Card.embedding` — cosine similarity, 1536 dimensions
- Unique constraints on all node unique properties

## Project Structure

```
MTG RAG Neo4j/
├── main.py                  # CLI entry point (ask, load, embed, detect-combos)
├── config.py                # All configuration loaded from .env
├── requirements.txt
├── .env.example
├── data/
│   └── cards.json           # Scryfall bulk data cache (created on first load)
├── graph/
│   ├── client.py            # Neo4j driver singleton
│   ├── schema.py            # Constraints and vector index setup
│   └── queries.py           # Cypher query library
├── ingestion/
│   ├── scryfall.py          # Fetch bulk card data from Scryfall API
│   ├── loader.py            # Parse cards and create Neo4j nodes/relationships
│   ├── embeddings.py        # Batch-embed oracle text via OpenAI
│   ├── spellbook.py         # Fetch combo data from Commander Spellbook API
│   └── combo_loader.py      # Parse combos and write Combo nodes to Neo4j
├── rag/
│   ├── retriever.py         # Hybrid retrieval: vector search + graph expansion
│   ├── chain.py             # GPT-4o-mini completion with retrieved context
│   └── prompts.py           # System and user prompt templates
├── combos/
│   └── detector.py          # Combo ingestion: fetches from Commander Spellbook API
└── tests/                   # Full test suite (all external I/O mocked)
```

## Running Tests

No live Neo4j or OpenAI connection required — all external I/O is mocked.

```bash
pytest tests/ -v
```

## Configuration Reference

All variables are read from `.env` via `python-dotenv`.

| Variable          | Default                   | Required | Description                                      |
|-------------------|---------------------------|----------|--------------------------------------------------|
| `NEO4J_URI`       |                           | Yes      | Neo4j Aura connection URI (`neo4j+s://...`)      |
| `NEO4J_USERNAME`  | `neo4j`                   | No       | Neo4j username                                   |
| `NEO4J_PASSWORD`  |                           | Yes      | Neo4j password                                   |
| `OPENAI_API_KEY`  |                           | Yes      | OpenAI API key                                   |
| `EMBEDDING_MODEL` | `text-embedding-3-small`  | No       | OpenAI embedding model                           |
| `LLM_MODEL`       | `gpt-4o-mini`             | No       | OpenAI chat model for answer generation          |
| `VECTOR_TOP_K`    | `10`                      | No       | Number of cards returned by vector search        |

## Attribution

Combo data provided by [Commander Spellbook](https://commanderspellbook.com/), an open-source project under the MIT license.

Card data provided by [Scryfall](https://scryfall.com/) under their [data policy](https://scryfall.com/docs/api/images).
