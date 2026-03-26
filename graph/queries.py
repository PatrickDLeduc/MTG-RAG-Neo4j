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
                c.id AS card_id,
                c.name AS card_name,
                c.oracle_text AS oracle_text,
                collect(DISTINCT k.name) AS keywords,
                collect(DISTINCT combo.description) AS combos,
                collect(DISTINCT partner.name) AS combo_cards
        """, ids=card_ids)
        return result.data()


def vector_search_cards_by_color(
    query_embedding: list[float],
    color: str,
    top_k: int = None,
) -> list[dict]:
    """Find cards most semantically similar to the query embedding, filtered by color."""
    k = top_k or config.VECTOR_TOP_K
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('card_embedding', $top_k, $embedding)
            YIELD node AS card, score
            WHERE (card)-[:HAS_COLOR]->(:Color {name: $color})
            RETURN card.id AS id,
                   card.name AS name,
                   card.oracle_text AS oracle_text,
                   score
        """, embedding=query_embedding, top_k=k, color=color)
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


def search_cards_by_name(names: list[str]) -> list[dict]:
    """Return cards whose name matches any of the given strings (case-insensitive).

    Matches against both the original name and a punctuation-stripped variant so
    that queries omitting hyphens or commas (e.g. 'Korvold Fae Cursed King')
    still find the card.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Card)
            WHERE any(n IN $names WHERE
                toLower(c.name) CONTAINS toLower(n)
                OR toLower(replace(replace(c.name, '-', ' '), ',', '')) CONTAINS toLower(n)
            )
            RETURN c.id AS id, c.name AS name, c.oracle_text AS oracle_text
        """, names=names)
        return result.data()


def get_combos_for_card(card_name: str) -> list[dict]:
    """Return all combos that include a specific card, with partner card names."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Card {name: $card_name})-[:PART_OF_COMBO]->(combo:Combo)
            OPTIONAL MATCH (combo)<-[:PART_OF_COMBO]-(partner:Card)
            WHERE partner.name <> $card_name
            RETURN
                combo.id AS combo_id,
                combo.description AS description,
                combo.combo_type AS combo_type,
                combo.popularity AS popularity,
                collect(DISTINCT partner.name) AS partner_cards
            ORDER BY combo.popularity DESC
        """, card_name=card_name)
        return result.data()
