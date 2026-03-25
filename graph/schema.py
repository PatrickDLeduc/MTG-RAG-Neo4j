from graph.client import get_driver


def setup_schema() -> None:
    """Create all constraints and indexes. Safe to run multiple times (IF NOT EXISTS)."""
    driver = get_driver()
    with driver.session() as session:
        _create_constraints(session)
        _create_vector_indexes(session)


def _create_constraints(session) -> None:
    constraints = [
        "CREATE CONSTRAINT card_id IF NOT EXISTS FOR (c:Card) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
        "CREATE CONSTRAINT cardtype_name IF NOT EXISTS FOR (t:CardType) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT subtype_name IF NOT EXISTS FOR (s:Subtype) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT color_name IF NOT EXISTS FOR (col:Color) REQUIRE col.name IS UNIQUE",
        "CREATE CONSTRAINT combo_id IF NOT EXISTS FOR (c:Combo) REQUIRE c.id IS UNIQUE",
    ]
    for cypher in constraints:
        session.run(cypher)


def _create_vector_indexes(session) -> None:
    # 1536 dimensions = text-embedding-3-small output size
    for cypher in [
        """
        CREATE VECTOR INDEX card_embedding IF NOT EXISTS
        FOR (c:Card) ON c.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """,
        """
        CREATE VECTOR INDEX keyword_embedding IF NOT EXISTS
        FOR (k:Keyword) ON k.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
        """,
    ]:
        session.run(cypher)
