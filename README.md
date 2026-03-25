# MTG RAG Neo4j

A Magic: The Gathering card combination assistant powered by hybrid RAG — combining vector search with Neo4j graph traversal to answer natural language questions about card synergies.

## Overview

Ask questions like *"What cards synergize with deathtouch?"* or *"Which cards work well with persist?"* and get grounded answers backed by real card data. The system embeds your query, finds semantically similar cards via vector search, expands the results through the card relationship graph (keywords, combos, types), and feeds the context to GPT-4o-mini to generate a focused answer.

Pure vector search misses structural relationships; pure graph traversal misses semantic similarity. The hybrid approach captures both.

## How It Works

```
INGESTION PIPELINE (run once)

  Scryfall API
      |
      v
  ingestion/scryfall.py  ──>  data/cards.json (local cache)
      |
      v
  ingestion/loader.py    ──>  Neo4j nodes: Card, Keyword, CardType, Subtype, Color
                                           + relationships: HAS_KEYWORD, HAS_TYPE, etc.
      |
      v
  ingestion/embeddings.py ─>  OpenAI text-embedding-3-small
                              ──>  Card.embedding stored in Neo4j (1536-dim, cosine)
      |
      v
  combos/detector.py     ──>  Combo nodes + PART_OF_COMBO relationships


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

# 3. Detect keyword synergy combos and write Combo nodes
python -m main detect-combos
```

The `load` command caches the Scryfall bulk data to `data/cards.json` so subsequent runs skip the download.

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

Run combo detection rules and write `Combo` nodes to Neo4j.

```bash
python -m main detect-combos
```

Currently detects:
- Deathtouch + Trample (1 damage kills any blocker; excess tramples over)
- Deathtouch + First Strike (kill any blocker before it can deal damage)
- Persist (infinite ETB loop enabler)

## Graph Schema

### Nodes

| Label      | Unique property | Key properties                                          |
|------------|----------------|---------------------------------------------------------|
| `Card`     | `id`           | `name`, `oracle_text`, `mana_cost`, `cmc`, `type_line`, `rarity`, `embedding` |
| `Keyword`  | `name`         | `embedding`                                             |
| `CardType` | `name`         |                                                         |
| `Subtype`  | `name`         |                                                         |
| `Color`    | `name`         |                                                         |
| `Combo`    | `id`           | `description`, `combo_type`                             |

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
│   └── embeddings.py        # Batch-embed oracle text via OpenAI
├── rag/
│   ├── retriever.py         # Hybrid retrieval: vector search + graph expansion
│   ├── chain.py             # GPT-4o-mini completion with retrieved context
│   └── prompts.py           # System and user prompt templates
├── combos/
│   └── detector.py          # Hardcoded combo detection rules
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
