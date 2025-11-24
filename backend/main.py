from typing import Optional, Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scoring import score_transcript


class ScoreRequest(BaseModel):
    transcript: str
    # Optional field if you know duration of speech in seconds.
    # If not provided, WPM is estimated neutrally.
    duration_seconds: Optional[float] = None


class ScoreResponse(BaseModel):
    overall_score: float
    words: int
    sentences: int
    duration_seconds: Optional[float]
    wpm: Optional[float]
    criteria: Any
    keyword_hits: Optional[Dict[str, Dict[str, bool]]] = None


app = FastAPI(
    title="Nirmaan AI – Communication Scoring API",
    description=(
        "Takes a self-introduction transcript and returns rubric-based scores "
        "using rule-based checks, NLP semantic similarity, and rubric weighting."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score_endpoint(body: ScoreRequest):
    try:
        result = score_transcript(
            transcript=body.transcript,
            duration_seconds=body.duration_seconds,
        )
        return result
    except Exception as e:
        import traceback
        print(f"Error in score_endpoint: {e}")
        print(traceback.format_exc())
        raise
