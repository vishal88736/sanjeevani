"""
WHO-grounded medical knowledge source.

Matches the translated English query against a small, curated table of
WHO fact-sheet topics (`WHO_FACT_SHEETS`), and — on a match — fetches
the live page from who.int and returns an excerpt as grounding context
plus the source URL, so answers can honestly say "according to WHO"
and the frontend can link back to the original page.

This is a keyword matcher over a hand-picked subset of WHO fact
sheets, not a full WHO search API (WHO doesn't publish one for this
content). Extend `WHO_FACT_SHEETS` with more topics/URLs as needed —
see the entries below for the expected shape.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBaseResult:
    context: str
    sources: list[str] = field(default_factory=list)
    is_grounded: bool = False


# topic -> (keywords to match in the English query, fact-sheet title, URL)
# Extend this table with more WHO fact sheets as needed. Find the
# canonical URL on https://www.who.int/news-room/fact-sheets
WHO_FACT_SHEETS: dict[str, tuple[list[str], str, str]] = {
    "malaria": (["malaria", "mosquito fever", "chills and fever"],
                "Malaria", "https://www.who.int/news-room/fact-sheets/detail/malaria"),
    "dengue": (["dengue", "break-bone fever"],
               "Dengue and severe dengue",
               "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue"),
    "tuberculosis": (["tuberculosis", "tb", "persistent cough", "coughing blood"],
                      "Tuberculosis", "https://www.who.int/news-room/fact-sheets/detail/tuberculosis"),
    "diarrhoea": (["diarrhoea", "diarrhea", "loose motion", "loose motions"],
                  "Diarrhoeal disease",
                  "https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease"),
    "pneumonia": (["pneumonia", "chest infection", "difficulty breathing child"],
                  "Pneumonia in children",
                  "https://www.who.int/news-room/fact-sheets/detail/pneumonia"),
    "diabetes": (["diabetes", "high blood sugar", "sugar disease"],
                 "Diabetes", "https://www.who.int/news-room/fact-sheets/detail/diabetes"),
    "hypertension": (["hypertension", "high blood pressure", "bp high"],
                      "Hypertension", "https://www.who.int/news-room/fact-sheets/detail/hypertension"),
    "anaemia": (["anaemia", "anemia", "low haemoglobin", "low hemoglobin"],
                "Anaemia", "https://www.who.int/news-room/fact-sheets/detail/anaemia"),
    "malnutrition": (["malnutrition", "undernutrition", "underweight child", "stunting"],
                      "Malnutrition", "https://www.who.int/news-room/fact-sheets/detail/malnutrition"),
    "hiv": (["hiv", "aids"],
            "HIV", "https://www.who.int/news-room/fact-sheets/detail/hiv-aids"),
    "cholera": (["cholera"],
                "Cholera", "https://www.who.int/news-room/fact-sheets/detail/cholera"),
    "typhoid": (["typhoid"],
                "Typhoid", "https://www.who.int/news-room/fact-sheets/detail/typhoid"),
    "maternal_health": (["pregnancy", "pregnant", "maternal", "childbirth", "labour pain"],
                         "Maternal mortality",
                         "https://www.who.int/news-room/fact-sheets/detail/maternal-mortality"),
    "mental_health": (["depression", "depressed", "anxiety", "anxious", "stress", "stressed",
                        "mental health", "suicidal", "sad", "hopeless", "worried"],
                       "Mental disorders",
                       "https://www.who.int/news-room/fact-sheets/detail/mental-disorders"),
    "covid19": (["covid", "coronavirus", "covid-19"],
                "COVID-19", "https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)"),
    "measles": (["measles"],
                "Measles", "https://www.who.int/news-room/fact-sheets/detail/measles"),
    "rabies": (["rabies", "dog bite"],
               "Rabies", "https://www.who.int/news-room/fact-sheets/detail/rabies"),
    "snakebite": (["snake bite", "snakebite"],
                  "Snakebite envenoming",
                  "https://www.who.int/news-room/fact-sheets/detail/snakebite-envenoming"),
}

_CACHE_TTL_SEC = 60 * 60 * 6  # 6 hours
_page_cache: dict[str, tuple[float, str]] = {}


def _match_topic(query: str) -> tuple[str, str] | None:
    """Returns (title, url) for the best-matching topic, or None."""
    query_lower = query.lower()
    best_topic = None
    best_score = 0
    for topic, (keywords, title, url) in WHO_FACT_SHEETS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_topic = (title, url)
    return best_topic if best_score > 0 else None


def _score_topic_name(topic_name: str, topic_key: str, keywords: list[str]) -> int:
    """Fuzzy-scores a free-text topic name (as suggested by Gemma, e.g.
    'Dengue fever' or 'malaria') against one WHO_FACT_SHEETS entry."""
    name_lower = topic_name.lower().strip()
    score = 0
    if topic_key in name_lower or name_lower in topic_key:
        score += 3
    for kw in keywords:
        if kw in name_lower or name_lower in kw:
            score += 1
    # word-level overlap as a softer fallback signal
    name_words = set(re.findall(r"[a-z]+", name_lower))
    key_words = set(re.findall(r"[a-z]+", topic_key)) | {w for kw in keywords for w in kw.split()}
    score += len(name_words & key_words)
    return score


def match_topics(topic_names: list[str], max_topics: int = 3) -> list[KnowledgeBaseResult]:
    """Fuzzy-matches each Gemma-suggested topic name (see
    model/llm/reasoning.py's retrieval-planning stage) against
    WHO_FACT_SHEETS, fetches the top `max_topics` distinct matches, and
    returns one KnowledgeBaseResult per matched topic. Topic names that
    don't match anything in the table are silently skipped — this is a
    curated-table lookup, not a general search."""
    scored: list[tuple[int, str, str]] = []  # (score, title, url)
    seen_urls: set[str] = set()

    for topic_name in topic_names:
        best = None
        best_score = 0
        for key, (keywords, title, url) in WHO_FACT_SHEETS.items():
            score = _score_topic_name(topic_name, key, keywords)
            if score > best_score:
                best_score = score
                best = (title, url)
        if best and best_score > 0 and best[1] not in seen_urls:
            scored.append((best_score, best[0], best[1]))
            seen_urls.add(best[1])

    scored.sort(key=lambda t: t[0], reverse=True)
    results: list[KnowledgeBaseResult] = []
    for _, title, url in scored[:max_topics]:
        try:
            excerpt = _fetch_fact_sheet_excerpt(url)
            context = f"WHO fact sheet — {title} ({url}):\n{excerpt}"
            results.append(KnowledgeBaseResult(context=context, sources=[url], is_grounded=True))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch WHO fact sheet %s: %s", url, exc)
            results.append(KnowledgeBaseResult(context="", sources=[url], is_grounded=False))

    return results


def _fetch_fact_sheet_excerpt(url: str, max_chars: int = 1200) -> str:
    """Fetches and lightly cleans a WHO fact-sheet page, cached in-memory."""
    cached = _page_cache.get(url)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]

    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=8, headers={"User-Agent": "SanjeevaniBot/1.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup
    text = re.sub(r"\s+", " ", main.get_text(separator=" ")).strip()
    excerpt = text[:max_chars]

    _page_cache[url] = (time.time(), excerpt)
    return excerpt


def retrieve_context(query: str) -> KnowledgeBaseResult:
    """Matches `query` against WHO_FACT_SHEETS and, on a match, fetches a
    live excerpt from who.int. Falls back to ungrounded if nothing
    matches or the fetch fails (e.g. no network access)."""
    match = _match_topic(query)
    if match is None:
        return KnowledgeBaseResult(context="", sources=[], is_grounded=False)

    title, url = match
    try:
        excerpt = _fetch_fact_sheet_excerpt(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch WHO fact sheet %s: %s", url, exc)
        return KnowledgeBaseResult(
            context="", sources=[url], is_grounded=False
        )

    context = f"WHO fact sheet — {title} ({url}):\n{excerpt}"
    return KnowledgeBaseResult(context=context, sources=[url], is_grounded=True)
