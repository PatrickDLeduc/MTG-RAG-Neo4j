from graph.client import get_driver


def detect_and_store() -> int:
    """Run all combo detection rules. Returns total number of card-combo relationships created/matched."""
    driver = get_driver()
    total = 0
    with driver.session() as session:
        total += _detect_deathtouch_trample(session)
        total += _detect_deathtouch_first_strike(session)
        total += _detect_persist_combo(session)
    return total


def _detect_deathtouch_trample(session) -> int:
    """Cards with both Deathtouch and Trample are a built-in synergy (1 deathtouch damage + trample rest)."""
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
    """Deathtouch + First Strike kills blockers before they can deal damage back."""
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
    """Cards with Persist are enablers of the Persist infinite ETB combo loop."""
    result = session.run("""
        MATCH (enabler:Card)-[:HAS_KEYWORD]->(k:Keyword {name: "Persist"})
        MERGE (combo:Combo {id: "infinite-persist-loop"})
        SET combo.description = "Persist infinite ETB: Persist creature + -1/-1 counter removal + sacrifice outlet = infinite enter-the-battlefield triggers",
            combo.combo_type = "infinite"
        MERGE (enabler)-[:PART_OF_COMBO {role: "enabler"}]->(combo)
        RETURN count(enabler) AS count
    """)
    return result.single()["count"]
