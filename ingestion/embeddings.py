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
        texts = [c["text"] or " " for c in batch]
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
