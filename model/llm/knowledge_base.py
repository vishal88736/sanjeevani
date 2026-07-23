"""
Multi-source medical knowledge base.

Matches the translated English query against curated tables of topics from:
- WHO (World Health Organization)
- NIH MedlinePlus
- First Aid emergency guides

On a match, fetches the live page (or uses hardcoded first aid text) and returns
an excerpt as grounding context plus the source URL. This allows answers to say
"according to WHO" or "according to NIH MedlinePlus" for maximum credibility.
"""

from __future__ import annotations

import logging
import re
import time
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBaseResult:
    context: str
    sources: list[str] = field(default_factory=list)
    is_grounded: bool = False
    source_name: str = ""



# topic -> (keywords to match in the English query, fact-sheet title, URL)
#
# 50 topics covering India's rural/semi-urban disease burden. Every
# URL below has been verified live on who.int. Keywords include both
# medical terms and colloquial phrases used in rural India.
#
# Find more at: https://www.who.int/news-room/fact-sheets
WHO_FACT_SHEETS: dict[str, tuple[list[str], str, str]] = {
    # ── Vector-borne & tropical infections ────────────────────────────
    "malaria": (["malaria", "mosquito fever", "chills and fever"],
                "Malaria", "https://www.who.int/news-room/fact-sheets/detail/malaria"),
    "dengue": (["dengue", "break-bone fever"],
               "Dengue and severe dengue",
               "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue"),
    "chikungunya": (["chikungunya", "joint pain fever", "mosquito joint pain"],
                     "Chikungunya",
                     "https://www.who.int/news-room/fact-sheets/detail/chikungunya"),
    "japanese_encephalitis": (["japanese encephalitis", "brain fever", "encephalitis"],
                               "Japanese encephalitis",
                               "https://www.who.int/news-room/fact-sheets/detail/japanese-encephalitis"),
    "lymphatic_filariasis": (["filariasis", "elephantiasis", "swollen leg", "swollen limb"],
                              "Lymphatic filariasis",
                              "https://www.who.int/news-room/fact-sheets/detail/lymphatic-filariasis"),
    "leishmaniasis": (["leishmaniasis", "kala-azar", "kala azar", "black fever"],
                       "Leishmaniasis",
                       "https://www.who.int/news-room/fact-sheets/detail/leishmaniasis"),
    "leptospirosis": (["leptospirosis", "rat fever", "weil disease", "flooding fever"],
                       "Leptospirosis",
                       "https://www.who.int/news-room/fact-sheets/detail/leptospirosis"),
    "schistosomiasis": (["schistosomiasis", "bilharzia", "blood in urine worms"],
                         "Schistosomiasis",
                         "https://www.who.int/news-room/fact-sheets/detail/schistosomiasis"),

    # ── Respiratory diseases ──────────────────────────────────────────
    "tuberculosis": (["tuberculosis", "tb", "persistent cough", "coughing blood",
                      "night sweats", "weight loss cough", "tb test"],
                      "Tuberculosis", "https://www.who.int/news-room/fact-sheets/detail/tuberculosis"),
    "pneumonia": (["pneumonia", "chest infection", "difficulty breathing child",
                   "lung infection", "breathing fast child", "fever cough child"],
                  "Pneumonia in children",
                  "https://www.who.int/news-room/fact-sheets/detail/pneumonia"),
    "asthma": (["asthma", "wheezing", "breathless", "difficulty breathing", "inhaler"],
               "Asthma", "https://www.who.int/news-room/fact-sheets/detail/asthma"),
    "copd": (["copd", "chronic bronchitis", "emphysema", "smoker cough",
              "breathing difficulty long term"],
             "Chronic obstructive pulmonary disease (COPD)",
             "https://www.who.int/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-(copd)"),
    "influenza": (["influenza", "flu", "seasonal flu", "body aches fever cold"],
                   "Influenza (seasonal)",
                   "https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)"),

    # ── Diarrhoeal & gut diseases ─────────────────────────────────────
    "diarrhoea": (["diarrhoea", "diarrhea", "loose motion", "loose motions",
                   "dehydration", "ors", "oral rehydration", "watery stool",
                   "stomach upset"],
                  "Diarrhoeal disease",
                  "https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease"),
    "cholera": (["cholera"],
                "Cholera", "https://www.who.int/news-room/fact-sheets/detail/cholera"),
    "typhoid": (["typhoid"],
                "Typhoid", "https://www.who.int/news-room/fact-sheets/detail/typhoid"),
    "soil_helminths": (["worms", "roundworm", "hookworm", "whipworm",
                        "stomach worms", "deworming"],
                       "Soil-transmitted helminth infections",
                       "https://www.who.int/news-room/fact-sheets/detail/soil-transmitted-helminth-infections"),

    # ── Maternal, newborn & child health ──────────────────────────────
    "maternal_health": (["pregnancy", "pregnant", "maternal", "childbirth", "labour pain"],
                         "Maternal mortality",
                         "https://www.who.int/news-room/fact-sheets/detail/maternal-mortality"),
    "newborn_health": (["newborn", "baby sick", "infant health", "neonatal",
                        "umbilical cord", "baby not feeding"],
                       "Newborn health",
                       "https://www.who.int/news-room/fact-sheets/detail/newborn-health"),
    "breastfeeding": (["breastfeeding", "breast milk", "baby feeding", "lactation",
                       "not enough milk"],
                      "Infant and young child feeding",
                      "https://www.who.int/news-room/fact-sheets/detail/infant-and-young-child-feeding"),
    "family_planning": (["family planning", "contraception", "birth control",
                         "unwanted pregnancy", "condom pill"],
                        "Family planning / contraception",
                        "https://www.who.int/news-room/fact-sheets/detail/family-planning-contraception"),
    "immunization": (["vaccination", "vaccine", "immunization", "immunisation",
                      "child vaccine schedule"],
                     "Immunization coverage",
                     "https://www.who.int/news-room/fact-sheets/detail/immunization-coverage"),

    # ── Non-communicable diseases (NCDs) ──────────────────────────────
    "diabetes": (["diabetes", "high blood sugar", "sugar disease"],
                 "Diabetes", "https://www.who.int/news-room/fact-sheets/detail/diabetes"),
    "hypertension": (["hypertension", "high blood pressure", "bp high"],
                      "Hypertension", "https://www.who.int/news-room/fact-sheets/detail/hypertension"),
    "cardiovascular": (["heart disease", "heart attack", "chest pain", "stroke",
                        "heart failure"],
                       "Cardiovascular diseases (CVDs)",
                       "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"),
    "cancer": (["cancer", "tumour", "tumor", "lump", "growth", "oncology"],
               "Cancer", "https://www.who.int/news-room/fact-sheets/detail/cancer"),
    "cervical_cancer": (["cervical cancer", "uterus cancer", "womb cancer",
                         "pap smear", "hpv"],
                        "Cervical cancer",
                        "https://www.who.int/news-room/fact-sheets/detail/cervical-cancer"),
    "epilepsy": (["epilepsy", "seizure", "seizures", "fits", "convulsion",
                  "falling sickness"],
                 "Epilepsy", "https://www.who.int/news-room/fact-sheets/detail/epilepsy"),
    "obesity": (["obesity", "overweight", "weight gain", "fat", "bmi"],
                "Obesity and overweight",
                "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight"),
    "chronic_kidney": (["kidney disease", "kidney failure", "dialysis",
                        "creatinine high", "kidney stone"],
                       "Chronic kidney disease",
                       "https://www.who.int/news-room/fact-sheets/detail/chronic-kidney-disease"),

    # ── Nutrition & blood disorders ───────────────────────────────────
    "anaemia": (["anaemia", "anemia", "low haemoglobin", "low hemoglobin"],
                "Anaemia", "https://www.who.int/news-room/fact-sheets/detail/anaemia"),
    "malnutrition": (["malnutrition", "undernutrition", "underweight child", "stunting"],
                      "Malnutrition", "https://www.who.int/news-room/fact-sheets/detail/malnutrition"),

    # ── Infectious diseases ───────────────────────────────────────────
    "hiv": (["hiv", "aids"],
            "HIV", "https://www.who.int/news-room/fact-sheets/detail/hiv-aids"),
    "hepatitis_a": (["hepatitis a", "jaundice", "yellow eyes", "liver infection",
                     "contaminated water jaundice"],
                    "Hepatitis A",
                    "https://www.who.int/news-room/fact-sheets/detail/hepatitis-a"),
    "hepatitis_b": (["hepatitis b", "liver disease", "carrier hepatitis",
                     "hepatitis vaccine"],
                    "Hepatitis B",
                    "https://www.who.int/news-room/fact-sheets/detail/hepatitis-b"),
    "measles": (["measles"],
                "Measles", "https://www.who.int/news-room/fact-sheets/detail/measles"),
    "covid19": (["covid", "coronavirus", "covid-19"],
                "COVID-19", "https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)"),
    "tetanus": (["tetanus", "lockjaw", "rust nail wound", "jaw stiff",
                 "muscle spasm wound"],
                "Tetanus",
                "https://www.who.int/news-room/fact-sheets/detail/tetanus"),
    "poliomyelitis": (["polio", "poliomyelitis", "paralysis child",
                       "leg weakness child", "polio drops"],
                      "Poliomyelitis",
                      "https://www.who.int/news-room/fact-sheets/detail/poliomyelitis"),
    "yellow_fever": (["yellow fever"],
                      "Yellow fever",
                      "https://www.who.int/news-room/fact-sheets/detail/yellow-fever"),

    # ── Neglected tropical diseases ───────────────────────────────────
    "leprosy": (["leprosy", "hansen disease", "skin patches numb",
                 "white patch no feeling"],
                "Leprosy",
                "https://www.who.int/news-room/fact-sheets/detail/leprosy"),

    # ── Mental health ─────────────────────────────────────────────────
    "mental_health": (["depression", "depressed", "anxiety", "anxious", "stress",
                       "stressed", "mental health", "suicidal", "sad", "hopeless",
                       "worried", "cannot sleep", "insomnia", "panic attack",
                       "no appetite sad", "self harm"],
                      "Mental disorders",
                      "https://www.who.int/news-room/fact-sheets/detail/mental-disorders"),

    # ── Bites & envenoming ────────────────────────────────────────────
    "rabies": (["rabies", "dog bite"],
               "Rabies", "https://www.who.int/news-room/fact-sheets/detail/rabies"),
    "snakebite": (["snake bite", "snakebite"],
                  "Snakebite envenoming",
                  "https://www.who.int/news-room/fact-sheets/detail/snakebite-envenoming"),

    # ── Water, food, environment & injuries ───────────────────────────
    "food_safety": (["food poisoning", "food safety", "vomiting after eating",
                     "stomach pain food", "contaminated food"],
                    "Food safety",
                    "https://www.who.int/news-room/fact-sheets/detail/food-safety"),
    "drinking_water": (["drinking water", "water safety", "dirty water",
                        "contaminated water", "water purification", "water borne"],
                       "Drinking-water",
                       "https://www.who.int/news-room/fact-sheets/detail/drinking-water"),
    "burns": (["burn", "burns", "scalding", "hot water burn", "fire burn", "acid burn"],
              "Burns", "https://www.who.int/news-room/fact-sheets/detail/burns"),
    "road_traffic": (["road accident", "road injury", "traffic accident",
                      "accident injury"],
                     "Road traffic injuries",
                     "https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries"),
    "drowning": (["drowning", "near drowning", "child fell water"],
                 "Drowning",
                 "https://www.who.int/news-room/fact-sheets/detail/drowning"),
    "heat_health": (["heat stroke", "heat exhaustion", "sun stroke",
                     "dehydration heat", "excessive heat"],
                    "Climate change, heat and health",
                    "https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health"),
}

MEDLINEPLUS_TOPICS: dict[str, tuple[list[str], str, str]] = {
    "diabetes": (["diabetes", "sugar", "high blood sugar"], "Diabetes", "https://medlineplus.gov/diabetes.html"),
    "heart_diseases": (["heart disease", "heart attack", "cardiovascular", "chest pain"], "Heart Diseases", "https://medlineplus.gov/heartdiseases.html"),
    "asthma": (["asthma", "wheezing", "breathless", "inhaler"], "Asthma", "https://medlineplus.gov/asthma.html"),
    "tuberculosis": (["tuberculosis", "tb", "persistent cough", "coughing blood"], "Tuberculosis", "https://medlineplus.gov/tuberculosis.html"),
    "malaria": (["malaria", "mosquito fever", "chills"], "Malaria", "https://medlineplus.gov/malaria.html"),
    "dengue": (["dengue", "break-bone fever"], "Dengue", "https://medlineplus.gov/dengue.html"),
    "pneumonia": (["pneumonia", "chest infection", "lung infection"], "Pneumonia", "https://medlineplus.gov/pneumonia.html"),
    "hepatitis": (["hepatitis", "jaundice", "liver infection"], "Hepatitis", "https://medlineplus.gov/hepatitis.html"),
    "hiv_aids": (["hiv", "aids"], "HIV/AIDS", "https://medlineplus.gov/hivaids.html"),
    "kidney_diseases": (["kidney disease", "kidney failure"], "Kidney Diseases", "https://medlineplus.gov/kidneydiseases.html"),
    "epilepsy": (["epilepsy", "seizure", "fits"], "Epilepsy", "https://medlineplus.gov/epilepsy.html"),
    "burns": (["burn", "burns", "scalding", "fire burn"], "Burns", "https://medlineplus.gov/burns.html"),
    "poisoning": (["poisoning", "poison", "toxic"], "Poisoning", "https://medlineplus.gov/poisoning.html"),
}

FIRST_AID_GUIDES: dict[str, tuple[list[str], str, str]] = {
    "choking": (["choking", "cannot breathe", "food stuck in throat"], "Choking First Aid",
                "If the person is choking and cannot breathe, cough, or speak: Perform the Heimlich maneuver. Stand behind them, wrap your arms around their waist, make a fist with one hand and place it just above their navel. Grab your fist with the other hand and give quick, upward thrusts into the stomach. Repeat until the object is dislodged. If they become unconscious, start CPR."),
    "severe_bleeding": (["bleeding heavily", "cut", "deep wound", "blood loss"], "Severe Bleeding First Aid",
                        "For severe bleeding: Apply direct, continuous pressure to the wound with a clean cloth or bandage. Do not remove the cloth if it soaks through; add more layers on top. Keep the injured area elevated above the heart if possible. Seek emergency medical help immediately."),
    "snakebite": (["snake bite", "snakebite", "bitten by a snake"], "Snakebite First Aid",
                  "If bitten by a snake: Keep the person calm and still to slow the spread of venom. Remove any tight clothing or jewelry near the bite before swelling starts. Keep the bitten limb at or slightly below the level of the heart. DO NOT cut the wound, attempt to suck out the venom, or apply a tourniquet. Get to a hospital immediately; antivenom is the only effective treatment."),
    "burns": (["severe burn", "acid burn", "fire burn"], "Burns First Aid",
              "For minor burns: Cool the burn with cool (not ice cold) running water for 10-15 minutes. For severe burns: Do not apply ointments, butter, or ice. Cover lightly with a clean, dry cloth. If clothing is stuck to the burn, do not try to peel it off. Seek emergency medical care immediately."),
    "drowning": (["drowning", "unconscious in water"], "Drowning First Aid",
                 "Get the person out of the water immediately. Check for breathing. If they are not breathing, begin CPR immediately (chest compressions and rescue breaths) if you are trained. Call for emergency help. Continue CPR until help arrives or the person starts breathing on their own."),
}

ALL_KNOWLEDGE_SOURCES = [
    ("WHO", WHO_FACT_SHEETS, True),
    ("NIH MedlinePlus", MEDLINEPLUS_TOPICS, True),
    ("First Aid Guide", FIRST_AID_GUIDES, False),
]

MEDQUAD_DB_PATH = Path(__file__).parent / "medquad.sqlite"

_CACHE_TTL_SEC = 60 * 60 * 6  # 6 hours
_page_cache: dict[str, tuple[float, str]] = {}


def _match_topic(query: str) -> tuple[str, str, str, bool] | None:
    """Returns (source_name, title, url_or_content, needs_fetch) for the best-matching topic, or None."""
    query_lower = query.lower()
    best_topic = None
    best_score = 0
    for source_name, source_dict, needs_fetch in ALL_KNOWLEDGE_SOURCES:
        for topic, (keywords, title, url_or_content) in source_dict.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_topic = (source_name, title, url_or_content, needs_fetch)
    return best_topic if best_score > 0 else None


def _search_medquad(query: str, max_results: int = 1) -> list[tuple[str, str, str]]:
    """Searches the local MedQuAD SQLite FTS5 database for matches.
    Returns a list of (source_name, title, content) tuples."""
    if not MEDQUAD_DB_PATH.exists():
        return []
        
    try:
        conn = sqlite3.connect(MEDQUAD_DB_PATH)
        cursor = conn.cursor()
        
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query).strip()
        if not clean_query:
            return []
            
        terms = [t for t in clean_query.split() if len(t) > 2]
        if not terms:
            return []
            
        match_query = " OR ".join(f'"{t}"*' for t in terms)
        
        cursor.execute('''
            SELECT qtype, question, answer
            FROM medquad 
            WHERE medquad MATCH ? 
            ORDER BY rank 
            LIMIT ?
        ''', (match_query, max_results))
        
        results = cursor.fetchall()
        conn.close()
        
        output = []
        for qtype, question, answer in results:
            title = f"{qtype} - {question}" if qtype else question
            output.append(("NIH MedQuAD", title, answer))
            
        return output
    except Exception as exc:
        logger.warning(f"MedQuAD search failed: {exc}")
        return []


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


def match_topics(topic_names: list[str], max_topics: int = 4) -> list[KnowledgeBaseResult]:
    """Fuzzy-matches each Gemma-suggested topic name (see
    model/llm/reasoning.py's retrieval-planning stage) against
    all knowledge sources, fetches the top `max_topics` distinct matches, and
    returns one KnowledgeBaseResult per matched topic."""
    scored: list[tuple[int, str, str, str, bool]] = []  # (score, source_name, title, url_or_content, needs_fetch)
    seen_keys: set[str] = set()

    for topic_name in topic_names:
        best = None
        best_score = 0
        for source_name, source_dict, needs_fetch in ALL_KNOWLEDGE_SOURCES:
            for key, (keywords, title, url_or_content) in source_dict.items():
                score = _score_topic_name(topic_name, key, keywords)
                if score > best_score:
                    best_score = score
                    best = (source_name, title, url_or_content, needs_fetch)
        if best and best_score > 0 and best[2] not in seen_keys:
            scored.append((best_score, best[0], best[1], best[2], best[3]))
            seen_keys.add(best[2])

    scored.sort(key=lambda t: t[0], reverse=True)
    results: list[KnowledgeBaseResult] = []
    for _, source_name, title, url_or_content, needs_fetch in scored[:max_topics]:
        try:
            if needs_fetch:
                excerpt = _fetch_fact_sheet_excerpt(url_or_content)
                context = f"{source_name} fact sheet — {title} ({url_or_content}):\n{excerpt}"
                results.append(KnowledgeBaseResult(context=context, sources=[url_or_content], is_grounded=True, source_name=source_name))
            else:
                context = f"{source_name} — {title}:\n{url_or_content}"
                results.append(KnowledgeBaseResult(context=context, sources=[], is_grounded=True, source_name=source_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s resource %s: %s", source_name, url_or_content, exc)
            if needs_fetch:
                results.append(KnowledgeBaseResult(context="", sources=[url_or_content], is_grounded=False, source_name=source_name))

    if len(results) < max_topics and MEDQUAD_DB_PATH.exists():
        for topic_name in topic_names:
            if len(results) >= max_topics:
                break
            medquad_matches = _search_medquad(topic_name, max_results=1)
            for source_name, title, content in medquad_matches:
                if len(results) < max_topics:
                    context = f"{source_name} — {title}:\n{content}"
                    results.append(KnowledgeBaseResult(context=context, sources=[], is_grounded=True, source_name=source_name))

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
    """Matches `query` against knowledge sources and, on a match, fetches a
    live excerpt or returns hardcoded text. Falls back to ungrounded if nothing
    matches or the fetch fails (e.g. no network access)."""
    match = _match_topic(query)
    if match is None:
        medquad_matches = _search_medquad(query, max_results=1)
        if medquad_matches:
            source_name, title, content = medquad_matches[0]
            context = f"{source_name} — {title}:\n{content}"
            return KnowledgeBaseResult(context=context, sources=[], is_grounded=True, source_name=source_name)
        return KnowledgeBaseResult(context="", sources=[], is_grounded=False)

    source_name, title, url_or_content, needs_fetch = match
    try:
        if needs_fetch:
            excerpt = _fetch_fact_sheet_excerpt(url_or_content)
            context = f"{source_name} fact sheet — {title} ({url_or_content}):\n{excerpt}"
            return KnowledgeBaseResult(context=context, sources=[url_or_content], is_grounded=True, source_name=source_name)
        else:
            context = f"{source_name} — {title}:\n{url_or_content}"
            return KnowledgeBaseResult(context=context, sources=[], is_grounded=True, source_name=source_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch %s resource %s: %s", source_name, url_or_content, exc)
        return KnowledgeBaseResult(
            context="", sources=[url_or_content] if needs_fetch else [], is_grounded=False, source_name=source_name
        )
