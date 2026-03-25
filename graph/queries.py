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
