import asyncio
from pathlib import Path

from graph.client import get_driver
from ingestion.spellbook import download_combos
from ingestion.combo_loader import load_combos


def detect_and_store(
    limit: int = 0,
    cache_path: Path = Path("data/combos.json"),
) -> tuple[int, bool]:
    """Fetch combos from Commander Spellbook and write to Neo4j.

    Also runs hardcoded keyword-synergy detection against the card graph
    (deathtouch+trample, deathtouch+first strike, persist loops).

    Returns (relationship_count, from_cache).
    """
    combos, from_cache = asyncio.run(download_combos(cache_path=cache_path))
    if limit:
        combos = combos[:limit]
    total = load_combos(combos)

    driver = get_driver()
    with driver.session() as session:
        total += _detect_keyword_synergies(session)
    return total, from_cache


def _detect_keyword_synergies(session) -> int:
    """Fallback: hardcoded keyword-matching rules (preserved, not called by default)."""
    total = 0
    total += _detect_deathtouch_trample(session)
    total += _detect_deathtouch_first_strike(session)
    total += _detect_persist_combo(session)
    return total


def _detect_deathtouch_trample(session) -> int:
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
    result = session.run("""
        MATCH (enabler:Card)-[:HAS_KEYWORD]->(k:Keyword {name: "Persist"})
        MERGE (combo:Combo {id: "infinite-persist-loop"})
        SET combo.description = "Persist infinite ETB: Persist creature + -1/-1 counter removal + sacrifice outlet = infinite enter-the-battlefield triggers",
            combo.combo_type = "infinite"
        MERGE (enabler)-[:PART_OF_COMBO {role: "enabler"}]->(combo)
        RETURN count(enabler) AS count
    """)
    return result.single()["count"]
