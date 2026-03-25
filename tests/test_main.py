from typer.testing import CliRunner
from unittest.mock import patch


def test_ask_command_prints_answer():
    runner = CliRunner()
    # main.py uses lazy imports inside commands, so patch the source modules directly
    with patch("rag.retriever.retrieve", return_value="Card: Lightning Bolt"), \
         patch("rag.chain.answer", return_value="Lightning Bolt deals 3 damage."):
        from main import app
        result = runner.invoke(app, ["ask", "What does Lightning Bolt do?"])

    assert result.exit_code == 0
    assert "Lightning Bolt deals 3 damage." in result.output


def test_load_command_exists():
    runner = CliRunner()
    # Just verify the command is registered — check help works
    from main import app
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "load" in result.output
