from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import math

import numpy as np
from sentence_transformers import SentenceTransformer
from language_tool_python import LanguageTool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------- Global models / tools (loaded once) ----------

_sentence_model: Optional[SentenceTransformer] = None
_language_tool: Optional[LanguageTool] = None
_sentiment_analyzer: Optional[SentimentIntensityAnalyzer] = None


def get_sentence_model() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        # small & fast, good for this case study
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def get_language_tool() -> LanguageTool:
    global _language_tool
    if _language_tool is None:
        _language_tool = LanguageTool("en-US")
    return _language_tool


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer


# ---------- Rubric config (based on Excel) ----------

@dataclass
class Criterion:
    id: str
    name: str
    description: str
    weight: float  # rubric weight / importance (will be normalized)
    max_points: float  # internal max raw points (for explanations)


# High-level rubric (matches “Overall Rubrics” section)
CRITERIA: List[Criterion] = [
    Criterion(
        id="content_structure",
        name="Content & Structure",
        description=(
            "Clear self-introduction with salutation, basic details (name, age, "
            "class/school, family), hobbies/interests, goals, a unique point or fun fact, "
            "and a closing in a logical order."
        ),
        weight=40.0,
        max_points=40.0,
    ),
    Criterion(
        id="speech_rate",
        name="Speech Rate (WPM)",
        description=(
            "Speech rate close to ideal range (~111–140 words per minute). "
            "Too fast or too slow reduces clarity."
        ),
        weight=10.0,
        max_points=10.0,
    ),
    Criterion(
        id="language_grammar",
        name="Language & Grammar",
        description=(
            "Low grammar error rate and good vocabulary richness (variety of words)."
        ),
        weight=20.0,
        max_points=20.0,
    ),
    Criterion(
        id="clarity",
        name="Clarity (Filler Words)",
        description=(
            "Clear delivery with minimal filler words (um, uh, like, you know, etc.)."
        ),
        weight=15.0,
        max_points=15.0,
    ),
    Criterion(
        id="engagement",
        name="Engagement / Positivity",
        description=(
            "Overall positive and enthusiastic tone; sounds interested, confident and grateful."
        ),
        weight=15.0,
        max_points=15.0,
    ),
]


# ---------- Basic text utilities ----------

FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "you know",
    "so",
    "actually",
    "basically",
    "right",
    "i mean",
    "well",
    "kinda",
    "sort of",
    "okay",
    "ok",
    "hmm",
    "ah",
}

KEYWORD_SEMANTIC_THRESHOLD = 0.68

SALUTATION_NORMAL = [
    "hi", "hello", "hey",
]

SALUTATION_GOOD = [
    "good morning", "good afternoon", "good evening", "good day",
    "hello everyone", "hi everyone", "hi all",
]

SALUTATION_EXCELLENT_PHRASES = [
    "i am excited to introduce",
    "i'm excited to introduce",
    "feeling great",
]

# “Must have” and “Good to have” items from rubric
KEYWORD_GROUPS_MUST = {
    "name": ["my name is", "myself", "i am"],
    "age": ["years old"],
    "school_class": [
        "class",
        "school",
        "studying in",
        "section",
        "grade",
        "standard",
        "public school",
    ],
    "family": [
        "family",
        "my mother",
        "my mom",
        "my father",
        "my dad",
        "my parents",
        "there are",
    ],
    "hobbies": [
        "hobby",
        "hobbies",
        "i like",
        "i love",
        "i enjoy",
        "playing",
        "play",
        "interest",
        "free time",
        "take wickets",
        "cricket",
        "reading",
        "dancing",
    ],
}

KEYWORD_GROUPS_GOOD = {
    "origin": ["i am from", "i'm from"],
    "ambition": ["my goal", "my dream", "ambition"],
    "unique": ["fun fact", "one thing about me"],
    "strengths": ["strength", "achievement", "achievements"],
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize_words(text: str) -> List[str]:
    # simple word tokenizer
    return [w for w in re.findall(r"\b\w+\b", text.lower())]


def count_sentences(text: str) -> int:
    # rough sentence count
    sents = re.split(r"[.!?]+", text.strip())
    return len([s for s in sents if s.strip()])


def split_into_sentences(text: str) -> List[str]:
    """Return a list of trimmed sentences for semantic keyword checks."""
    raw = re.split(r"[.!?]+", text.strip())
    return [s.strip() for s in raw if s.strip()]


# ---------- Similarity ----------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_similarity(a: str, b: str) -> float:
    model = get_sentence_model()
    embeds = model.encode([a, b])
    sim = cosine_similarity(embeds[0], embeds[1])
    # map [-1, 1] -> [0, 1]
    return (sim + 1.0) / 2.0


# ---------- Individual metric scorers ----------

def score_salutation(text: str) -> Tuple[float, str]:
    """Return points out of 5 + feedback according to rubric."""
    t = normalize_text(text)

    score = 0.0
    level = "No salutation"

    if any(p in t for p in SALUTATION_EXCELLENT_PHRASES):
        score = 5.0
        level = "Excellent"
    elif any(p in t for p in SALUTATION_GOOD):
        score = 4.0
        level = "Good"
    elif any(p in t for p in SALUTATION_NORMAL):
        score = 2.0
        level = "Normal"

    feedback = f"Salutation level: {level} (score {score}/5)."
    return score, feedback


def score_keyword_presence(text: str) -> Tuple[float, str, Dict[str, Dict[str, bool]]]:
    """
    Implements 'Key word presence' from rubric.

    Must-haves: each 4 points (5 items => 20 points)
    Good-to-have: each 2 points (4 items => 8 points)
    Total possible: 28; we scale to 30 as in rubric.
    """
    normalized_text = normalize_text(text)
    sentences = split_into_sentences(text)
    sentence_model = get_sentence_model()
    sentence_embeddings = (
        sentence_model.encode(sentences) if sentences else np.array([])
    )

    def has_semantic_match(phrases: List[str]) -> bool:
        if sentence_embeddings.size == 0 or not phrases:
            return False
        phrase_embeddings = sentence_model.encode(phrases)
        for phrase_vec in phrase_embeddings:
            for sent_vec in sentence_embeddings:
                if cosine_similarity(sent_vec, phrase_vec) >= KEYWORD_SEMANTIC_THRESHOLD:
                    return True
        return False

    must_points = 0.0
    must_items_hit = []
    must_hits_map: Dict[str, bool] = {}
    for label, phrases in KEYWORD_GROUPS_MUST.items():
        literal_hit = any(p in normalized_text for p in phrases)
        semantic_hit = False
        if not literal_hit:
            semantic_hit = has_semantic_match(phrases)
        hit = literal_hit or semantic_hit
        must_hits_map[label] = hit
        if hit:
            must_points += 4.0
            must_items_hit.append(label)

    good_points = 0.0
    good_items_hit = []
    good_hits_map: Dict[str, bool] = {}
    for label, phrases in KEYWORD_GROUPS_GOOD.items():
        literal_hit = any(p in normalized_text for p in phrases)
        semantic_hit = False
        if not literal_hit:
            semantic_hit = has_semantic_match(phrases)
        hit = literal_hit or semantic_hit
        good_hits_map[label] = hit
        if hit:
            good_points += 2.0
            good_items_hit.append(label)

    raw_points = must_points + good_points  # max 28
    scaled_points = min(raw_points * (30.0 / 28.0), 30.0)

    feedback_parts = []
    feedback_parts.append(f"Must-have items hit: {', '.join(must_items_hit) or 'none'} "
                          f"({must_points}/20 raw).")
    feedback_parts.append(f"Good-to-have items hit: {', '.join(good_items_hit) or 'none'} "
                          f"({good_points}/8 raw).")
    feedback = " ".join(feedback_parts)

    keyword_details = {
        "must": must_hits_map,
        "good": good_hits_map,
    }

    return scaled_points, feedback, keyword_details


def score_flow(text: str) -> Tuple[float, str]:
    """
    Very simple approximation of 'Flow' according to order:
    Salutation -> basic details -> additional details -> closing.
    5 points if mainly followed, 0 otherwise.
    """
    t = normalize_text(text)

    has_salutation = any(p in t for p in SALUTATION_NORMAL + SALUTATION_GOOD)
    has_name = any(p in t for p in KEYWORD_GROUPS_MUST["name"])
    has_closing = "thank you" in t or "thanks for listening" in t

    score = 0.0
    if has_salutation and has_name and has_closing:
        score = 5.0

    feedback = (
        f"Flow check: salutation={has_salutation}, basic details={has_name}, "
        f"closing={has_closing}. Score {score}/5."
    )
    return score, feedback


def score_speech_rate(words: int, duration_seconds: Optional[float]) -> Tuple[float, str, float]:
    """
    Score speech rate out of 10 using rubric ranges.

    If no duration is provided, we estimate using 120 WPM as a neutral default
    (so the candidate is not penalized).
    """
    if words <= 0:
        return 0.0, "No words found – cannot compute speech rate.", 0.0

    if duration_seconds is None or duration_seconds <= 0:
        # assume neutral rate
        wpm = 120.0
        est_note = " (estimated using default 120 WPM as duration not provided)"
    else:
        wpm = words / (duration_seconds / 60.0)
        est_note = ""

    # rubric ranges:
    # >161 Too Fast -> 2
    # 141–160 Fast -> 6
    # 111–140 Ideal -> 10
    # 81–110 Slow -> 6
    # <80 Too Slow -> 2
    if wpm > 161:
        pts = 2.0
        level = "Too fast"
    elif 141 <= wpm <= 160:
        pts = 6.0
        level = "Fast"
    elif 111 <= wpm <= 140:
        pts = 10.0
        level = "Ideal"
    elif 81 <= wpm <= 110:
        pts = 6.0
        level = "Slow"
    else:
        pts = 2.0
        level = "Too slow"

    feedback = f"Speech rate: {wpm:.1f} WPM → {level} (score {pts}/10){est_note}."
    return pts, feedback, wpm


def score_grammar_and_vocab(words: int, text: str) -> Tuple[float, float, str]:
    """
    Return (grammar_points out of 10, vocab_points out of 10, feedback).
    Grammar score uses LanguageTool's error count per 100 words.
    Vocabulary richness uses Type-Token Ratio (TTR).
    """
    tool = get_language_tool()
    matches = tool.check(text)
    error_count = len(matches)
    if words == 0:
        errors_per_100 = 0.0
    else:
        errors_per_100 = (error_count / max(words, 1)) * 100.0

    # Grammar score per rubric: 1 - min(errors_per_100/10, 1)
    grammar_score_0_to_1 = 1.0 - min(errors_per_100 / 10.0, 1.0)
    grammar_points = grammar_score_0_to_1 * 10.0

    tokens = tokenize_words(text)
    distinct_tokens = len(set(tokens))
    ttr = distinct_tokens / max(len(tokens), 1)

    # TTR → points mapping (based on ranges in sheet)
    if ttr >= 0.9:
        vocab_points = 10.0
    elif 0.7 <= ttr < 0.9:
        vocab_points = 8.0
    elif 0.5 <= ttr < 0.7:
        vocab_points = 6.0
    elif 0.3 <= ttr < 0.5:
        vocab_points = 4.0
    else:
        vocab_points = 2.0

    feedback = (
        f"Grammar: {error_count} errors (~{errors_per_100:.1f} per 100 words) "
        f"→ {grammar_points:.1f}/10. "
        f"Vocabulary richness (TTR={ttr:.2f}) → {vocab_points:.1f}/10."
    )

    return grammar_points, vocab_points, feedback


def score_filler_words(words: int, text: str) -> Tuple[float, str, int, float]:
    """
    Filler word rate (number of filler tokens / total words * 100)
    -> points according to rubric:
        0–3  -> 15
        4–6  -> 12
        7–9  -> 9
        10–12 -> 6
        >=13  -> 3
    """
    t = normalize_text(text)
    tokens = tokenize_words(t)

    count = 0
    for w in tokens:
        if w in FILLER_WORDS:
            count += 1

    if words == 0:
        rate = 0.0
    else:
        rate = (count / words) * 100.0

    if rate <= 3:
        pts = 15.0
        band = "Excellent"
    elif 4 <= rate <= 6:
        pts = 12.0
        band = "Good"
    elif 7 <= rate <= 9:
        pts = 9.0
        band = "Moderate"
    elif 10 <= rate <= 12:
        pts = 6.0
        band = "High"
    else:
        pts = 3.0
        band = "Very high"

    feedback = (
        f"Filler words: {count} (~{rate:.1f}% of words) → {band} (score {pts}/15)."
    )
    return pts, feedback, count, rate


def score_sentiment(text: str) -> Tuple[float, str, float]:
    """
    Sentiment-based engagement score using VADER compound score:
        >=0.9   -> 15
        0.7–0.89 -> 12
        0.5–0.69 -> 9
        0.3–0.49 -> 6
        <0.3     -> 3
    """
    analyzer = get_sentiment_analyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.9:
        pts = 15.0
        band = "Very positive"
    elif 0.7 <= compound < 0.9:
        pts = 12.0
        band = "Positive"
    elif 0.5 <= compound < 0.7:
        pts = 9.0
        band = "Somewhat positive"
    elif 0.3 <= compound < 0.5:
        pts = 6.0
        band = "Neutral / mixed"
    else:
        pts = 3.0
        band = "Low positivity"

    feedback = f"Sentiment compound={compound:.2f} → {band} (score {pts}/15)."
    return pts, feedback, compound


# ---------- Main scoring pipeline ----------

def score_transcript(
    transcript: str,
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    End-to-end scoring:
    - Content & Structure (salutation + keywords + flow + semantic similarity)
    - Speech rate
    - Language & Grammar (errors + TTR)
    - Clarity (filler rate)
    - Engagement (sentiment)
    """

    text = transcript.strip()
    norm_text = normalize_text(text)
    tokens = tokenize_words(text)
    word_count = len(tokens)
    sentence_count = count_sentences(text)

    # --- Content & Structure ---
    sal_pts, sal_fb = score_salutation(text)
    kw_pts, kw_fb, kw_details = score_keyword_presence(text)
    flow_pts, flow_fb = score_flow(text)
    # semantic similarity vs rubric description
    content_criterion = next(c for c in CRITERIA if c.id == "content_structure")
    sem_sim = semantic_similarity(text, content_criterion.description)

    # combine sub-parts: scale to max_points=40
    # weights: sal 5, keywords 30, flow 5 (already on those scales)
    # plus semantic similarity (0–1) contributes up to +5 bonus
    content_raw_points = sal_pts + kw_pts + flow_pts + (sem_sim * 5.0)
    # cap at 40
    content_points = min(content_raw_points, content_criterion.max_points)

    content_feedback = (
        f"{sal_fb} {kw_fb} {flow_fb} "
        f"Semantic similarity to ideal content description: {sem_sim:.2f}."
    )

    # --- Speech rate (may be estimated) ---
    speech_criterion = next(c for c in CRITERIA if c.id == "speech_rate")
    speech_pts, speech_fb, wpm = score_speech_rate(word_count, duration_seconds)

    # --- Language & Grammar ---
    lang_criterion = next(c for c in CRITERIA if c.id == "language_grammar")
    grammar_pts, vocab_pts, lang_fb = score_grammar_and_vocab(word_count, text)
    # total out of 20
    lang_points = min(grammar_pts + vocab_pts, lang_criterion.max_points)

    # --- Clarity (filler words) ---
    clarity_criterion = next(c for c in CRITERIA if c.id == "clarity")
    clarity_pts, clarity_fb, filler_count, filler_rate = score_filler_words(
        word_count, text
    )
    clarity_points = min(clarity_pts, clarity_criterion.max_points)

    # --- Engagement (sentiment) ---
    engage_criterion = next(c for c in CRITERIA if c.id == "engagement")
    engagement_pts, engagement_fb, compound = score_sentiment(text)
    engagement_points = min(engagement_pts, engage_criterion.max_points)

    # --- Combine with rubric weights → 0–100 ---
    # We map each criterion's points to [0,1] by dividing by its max_points,
    # then multiply by its rubric weight.
    weighted_sum = 0.0
    total_weights = sum(c.weight for c in CRITERIA)

    def normalize_and_weight(points: float, criterion: Criterion) -> float:
        if criterion.max_points <= 0:
            return 0.0
        ratio = points / criterion.max_points
        return ratio * criterion.weight

    weighted_sum += normalize_and_weight(content_points, content_criterion)
    weighted_sum += normalize_and_weight(speech_pts, speech_criterion)
    weighted_sum += normalize_and_weight(lang_points, lang_criterion)
    weighted_sum += normalize_and_weight(clarity_points, clarity_criterion)
    weighted_sum += normalize_and_weight(engagement_points, engage_criterion)

    overall_score = (weighted_sum / total_weights) * 100.0 if total_weights > 0 else 0.0

    # --- Build per-criterion result list ---
    criteria_results = [
        {
            "id": "content_structure",
            "name": content_criterion.name,
            "points": round(content_points, 2),
            "max_points": content_criterion.max_points,
            "weight": content_criterion.weight,
            "normalized": round(content_points / content_criterion.max_points, 3)
            if content_criterion.max_points
            else 0.0,
            "feedback": content_feedback,
            "details": {
                "must_hits": kw_details.get("must", {}),
                "good_hits": kw_details.get("good", {}),
            },
        },
        {
            "id": "speech_rate",
            "name": speech_criterion.name,
            "points": round(speech_pts, 2),
            "max_points": speech_criterion.max_points,
            "weight": speech_criterion.weight,
            "normalized": round(speech_pts / speech_criterion.max_points, 3)
            if speech_criterion.max_points
            else 0.0,
            "feedback": speech_fb,
            "details": {"wpm": round(wpm, 2)},
        },
        {
            "id": "language_grammar",
            "name": lang_criterion.name,
            "points": round(lang_points, 2),
            "max_points": lang_criterion.max_points,
            "weight": lang_criterion.weight,
            "normalized": round(lang_points / lang_criterion.max_points, 3)
            if lang_criterion.max_points
            else 0.0,
            "feedback": lang_fb,
        },
        {
            "id": "clarity",
            "name": clarity_criterion.name,
            "points": round(clarity_points, 2),
            "max_points": clarity_criterion.max_points,
            "weight": clarity_criterion.weight,
            "normalized": round(clarity_points / clarity_criterion.max_points, 3)
            if clarity_criterion.max_points
            else 0.0,
            "feedback": clarity_fb,
            "details": {
                "filler_count": filler_count,
                "filler_rate": round(filler_rate, 2),
            },
        },
        {
            "id": "engagement",
            "name": engage_criterion.name,
            "points": round(engagement_points, 2),
            "max_points": engage_criterion.max_points,
            "weight": engage_criterion.weight,
            "normalized": round(engagement_points / engage_criterion.max_points, 3)
            if engage_criterion.max_points
            else 0.0,
            "feedback": engagement_fb,
            "details": {"sentiment_compound": round(compound, 3)},
        },
    ]

    return {
        "overall_score": round(overall_score, 2),
        "words": word_count,
        "sentences": sentence_count,
        "duration_seconds": duration_seconds,
        "wpm": round(wpm, 2) if word_count > 0 else None,
        "criteria": criteria_results,
        "keyword_hits": kw_details,
    }
