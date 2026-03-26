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
def detect_combos(
    cache: str = typer.Option("data/combos.json", help="Path to combo data cache"),
    limit: int = typer.Option(0, help="Limit combos to load (0 = all, sorted by popularity)"),
):
    """Fetch combos from Commander Spellbook and write to Neo4j."""
    from pathlib import Path
    from combos.detector import detect_and_store
    typer.echo("Loading combos...")
    count, from_cache = detect_and_store(limit=limit, cache_path=Path(cache))
    source = f"cache ({cache})" if from_cache else "Commander Spellbook API"
    typer.echo(f"Done. Source: {source}. Created/updated {count} combo-card relationships.")


if __name__ == "__main__":
    app()
