import re
from openai import OpenAI
from graph.queries import (
    vector_search_cards,
    vector_search_cards_by_color,
    expand_from_cards,
    search_cards_by_name,
)
import config

_client = OpenAI(api_key=config.OPENAI_API_KEY)

_PUNCT_RE = re.compile(r"[,\-–—/\\]+")

_COLOR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bcolorless\b', re.IGNORECASE), 'Colorless'),
    (re.compile(r'\bwhite\b',     re.IGNORECASE), 'White'),
    (re.compile(r'\bblue\b',      re.IGNORECASE), 'Blue'),
    (re.compile(r'\bblack\b',     re.IGNORECASE), 'Black'),
    (re.compile(r'\bred\b',       re.IGNORECASE), 'Red'),
    (re.compile(r'\bgreen\b',     re.IGNORECASE), 'Green'),
]


def _normalize_name(s: str) -> str:
    return _PUNCT_RE.sub(" ", s).strip()


def _extract_card_name_candidates(query: str) -> list[str]:
    """Extract potential card names from the query string."""
    quoted = re.findall(r'"([^"]+)"', query)
    titled = re.findall(r"\b(?:[A-Z][a-z']+(?:\s+[A-Z][a-z']+)+)\b", query)
    candidates = quoted + titled
    normalized = [_normalize_name(c) for c in candidates]
    return list(dict.fromkeys(candidates + normalized))


def _detect_color_filter(query: str) -> str | None:
    """Return a canonical color name if the query mentions a specific color, else None."""
    for pattern, color_name in _COLOR_PATTERNS:
        if pattern.search(query):
            return color_name
    return None


def retrieve(query: str) -> str:
    """Name lookup (priority) + vector search → graph expand → formatted context string."""
    embedding = _embed_query(query)
    color_filter = _detect_color_filter(query)

    # Name search runs FIRST — exact matches take priority in LLM context
    name_candidates = _extract_card_name_candidates(query)
    name_cards: list[dict] = []
    seen_ids: set[str] = set()
    if name_candidates:
        for row in search_cards_by_name(name_candidates):
            if row["id"] not in seen_ids:
                name_cards.append(row)
                seen_ids.add(row["id"])

    # Vector search supplements with semantically related cards
    if color_filter:
        vector_cards = vector_search_cards_by_color(embedding, color_filter)
    else:
        vector_cards = vector_search_cards(embedding)

    # De-duplicate: skip vector results already found by name search
    filtered_vector = [c for c in vector_cards if c["id"] not in seen_ids]

    # Name matches come FIRST so they appear first in LLM context
    all_card_ids = [c["id"] for c in name_cards] + [c["id"] for c in filtered_vector]
    id_order = {card_id: idx for idx, card_id in enumerate(all_card_ids)}

    expanded = expand_from_cards(all_card_ids)
    expanded.sort(key=lambda card: id_order.get(card.get("card_id"), len(all_card_ids)))
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
