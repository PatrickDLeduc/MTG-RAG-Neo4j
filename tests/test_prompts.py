def test_user_prompt_includes_context_and_question():
    from rag.prompts import USER_PROMPT_TEMPLATE
    result = USER_PROMPT_TEMPLATE.format(context="Card: Foo\n  Keywords: Flying", question="What flies?")
    assert "Card: Foo" in result
    assert "What flies?" in result


def test_system_prompt_instructs_grounding():
    from rag.prompts import SYSTEM_PROMPT
    # Must instruct the LLM to use only provided context (prevent hallucination)
    assert "context" in SYSTEM_PROMPT.lower() or "only" in SYSTEM_PROMPT.lower()
