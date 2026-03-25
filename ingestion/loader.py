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
    # The em-dash (—) separates main types from subtypes
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
        WITH card, sub_name
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
