from neo4j import GraphDatabase
import config

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )
    return _driver


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
