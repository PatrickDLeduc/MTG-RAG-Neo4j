from openai import OpenAI
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def answer(question: str, context: str) -> str:
    """Generate a grounded answer using the retrieved context."""
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    response = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
