from openai import OpenAI
from graph.queries import vector_search_cards, expand_from_cards
import config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def retrieve(query: str) -> str:
    """Embed query → vector search → graph expand → formatted context string."""
    embedding = _embed_query(query)
    cards = vector_search_cards(embedding)
    card_ids = [c["id"] for c in cards]
    expanded = expand_from_cards(card_ids)
    return _format_context(expanded)


def _embed_query(query: str) -> list[float]:
    client = _get_client()
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def _format_context(expanded: list[dict]) -> str:
    sections = []
    for card in expanded:
        lines = [f"Card: {card['card_name']}"]
        if card["oracle_text"]:
            lines.append(f"  Text: {card['oracle_text']}")
        if card["keywords"]:
            lines.append(f"  Keywords: {', '.join(card['keywords'])}")
        if card["combos"]:
            for combo in card["combos"]:
                lines.append(f"  Combo: {combo}")
        if card["combo_cards"]:
            lines.append(f"  Combo partners: {', '.join(card['combo_cards'])}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
