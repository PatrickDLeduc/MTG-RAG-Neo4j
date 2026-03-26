import re
from openai import OpenAI
from graph.queries import vector_search_cards, expand_from_cards, search_cards_by_name
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)

_PUNCT_RE = re.compile(r"[,\-–—/\\]+")


def _normalize_name(s: str) -> str:
    return _PUNCT_RE.sub(" ", s).strip()


def _extract_card_name_candidates(query: str) -> list[str]:
    """Extract potential card names from the query string."""
    quoted = re.findall(r'"([^"]+)"', query)
    titled = re.findall(r"\b(?:[A-Z][a-z']+(?:\s+[A-Z][a-z']+)+)\b", query)
    candidates = quoted + titled
    normalized = [_normalize_name(c) for c in candidates]
    return list(dict.fromkeys(candidates + normalized))


def retrieve(query: str) -> str:
    """Embed query → vector search + name lookup → graph expand → formatted context string."""
    embedding = _embed_query(query)
    vector_cards = vector_search_cards(embedding)
    seen_ids = {c["id"] for c in vector_cards}

    # Name-based fallback: find any cards explicitly named in the query
    name_candidates = _extract_card_name_candidates(query)
    name_cards = []
    if name_candidates:
        for row in search_cards_by_name(name_candidates):
            if row["id"] not in seen_ids:
                name_cards.append(row)
                seen_ids.add(row["id"])

    all_card_ids = [c["id"] for c in vector_cards] + [c["id"] for c in name_cards]
    expanded = expand_from_cards(all_card_ids)
    return _format_context(expanded)


def _embed_query(query: str) -> list[float]:
    response = _client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
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
