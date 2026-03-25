from openai import OpenAI
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def answer(question: str, context: str) -> str:
    """Generate a grounded answer using the retrieved context."""
    client = _get_client()
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
