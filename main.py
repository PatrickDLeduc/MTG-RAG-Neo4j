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
