from unittest.mock import patch, MagicMock
import importlib
import sys


def test_get_driver_returns_driver_instance():
    # Remove cached modules to start fresh
    if "graph.client" in sys.modules:
        del sys.modules["graph.client"]

    with patch("neo4j.GraphDatabase") as mock_gdb, \
         patch("config.NEO4J_URI", "neo4j://localhost:7687"), \
         patch("config.NEO4J_USERNAME", "neo4j"), \
         patch("config.NEO4J_PASSWORD", "password"):
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        from graph.client import get_driver
        driver = get_driver()

        assert driver is mock_driver
        mock_gdb.driver.assert_called_once()


def test_get_driver_returns_same_instance_on_second_call():
    # Remove cached modules to start fresh
    if "graph.client" in sys.modules:
        del sys.modules["graph.client"]

    with patch("neo4j.GraphDatabase") as mock_gdb, \
         patch("config.NEO4J_URI", "neo4j://localhost:7687"), \
         patch("config.NEO4J_USERNAME", "neo4j"), \
         patch("config.NEO4J_PASSWORD", "password"):
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        from graph.client import get_driver
        d1 = get_driver()
        d2 = get_driver()

        assert d1 is d2
        assert mock_gdb.driver.call_count == 1  # only created once


def test_close_driver_resets_singleton():
    # Remove cached modules to start fresh
    if "graph.client" in sys.modules:
        del sys.modules["graph.client"]

    with patch("neo4j.GraphDatabase") as mock_gdb, \
         patch("config.NEO4J_URI", "neo4j://localhost:7687"), \
         patch("config.NEO4J_USERNAME", "neo4j"), \
         patch("config.NEO4J_PASSWORD", "password"):
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        from graph.client import get_driver, close_driver
        get_driver()
        close_driver()
        mock_driver.close.assert_called_once()

        get_driver()
        assert mock_gdb.driver.call_count == 2  # recreated after close
