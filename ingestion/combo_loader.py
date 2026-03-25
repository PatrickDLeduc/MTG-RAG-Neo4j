from graph.client import get_driver

BATCH_SIZE = 500


def load_combos(combos: list[dict]) -> int:
    """Parse Spellbook combo variants and write to Neo4j. Returns relationship count."""
    driver = get_driver()
    parsed = [_parse_variant(c) for c in combos]
    total = 0
    for i in range(0, len(parsed), BATCH_SIZE):
        batch = parsed[i:i + BATCH_SIZE]
        with driver.session() as session:
            session.execute_write(_write_combo_nodes, batch)
            total += session.execute_write(_write_combo_relationships, batch)
    return total


def _parse_variant(variant: dict) -> dict:
    description = (variant.get("description") or "")[:500]
    return {
        "spellbook_id": variant["id"],
        "description": description,
        "color_identity": variant.get("identity") or "",
        "popularity": variant.get("popularity") or 0,
        "combo_type": _infer_combo_type(description),
        "card_names": [u["card"]["name"] for u in variant.get("uses", [])],
    }


def _infer_combo_type(description: str) -> str:
    text_lower = description.lower()
    if "infinite" in text_lower:
        return "infinite"
    if "win the game" in text_lower:
        return "wincon"
    return "synergy"


def _write_combo_nodes(tx, batch: list[dict]) -> None:
    tx.run("""
        UNWIND $combos AS combo
        MERGE (cb:Combo {id: combo.spellbook_id})
        SET cb.description = combo.description,
            cb.color_identity = combo.color_identity,
            cb.popularity = combo.popularity,
            cb.combo_type = combo.combo_type,
            cb.source = 'spellbook'
    """, combos=batch)


def _write_combo_relationships(tx, batch: list[dict]) -> int:
    result = tx.run("""
        UNWIND $combos AS combo
        UNWIND combo.card_names AS card_name
        MATCH (c:Card {name: card_name})
        MATCH (cb:Combo {id: combo.spellbook_id})
        MERGE (c)-[r:PART_OF_COMBO]->(cb)
        RETURN count(r) AS count
    """, combos=batch)
    record = result.single()
    return record["count"] if record else 0
