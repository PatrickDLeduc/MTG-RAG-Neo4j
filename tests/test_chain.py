from unittest.mock import MagicMock, patch
import importlib


def test_answer_calls_openai_with_context_and_question():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Kitchen Finks combos with Melira."

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = mock_response

    from rag import chain
    # Replace the module-level _client with a mock
    chain._client = mock_openai
    result = chain.answer("What combos with Persist?", "Card: Kitchen Finks\n  Keywords: Persist")

    assert result == "Kitchen Finks combos with Melira."
    call_args = mock_openai.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert any("Kitchen Finks" in m["content"] for m in messages)
    assert any("What combos with Persist?" in m["content"] for m in messages)


def test_answer_uses_low_temperature():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer"
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = mock_response

    from rag import chain
    # Replace the module-level _client with a mock
    chain._client = mock_openai
    chain.answer("Q", "context")

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] <= 0.2  # low temperature for factual answers
