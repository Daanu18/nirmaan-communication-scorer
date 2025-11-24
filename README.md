# Nirmaan AI – Communication Scoring Tool

FastAPI + vanilla JS application that scores self-introduction transcripts for the **Nirmaan AI Intern Case Study**. The tool ingests a transcript, evaluates it against the official rubric using rule-based checks plus NLP features, and returns:

- Overall score on a 0–100 scale
- Detailed per-criterion scores, weights, and textual feedback
- Keyword hit map for the must-have / good-to-have rubric items

The frontend lets evaluators paste a transcript, optionally add duration (seconds) for accurate WPM, and immediately visualize the rubric results.

---

## Architecture

| Layer      | Stack / Notes                                                                                                          |
|------------|------------------------------------------------------------------------------------------------------------------------|
| Backend    | FastAPI, Pydantic, SentenceTransformers, LanguageTool, VADER. Endpoint `/score` accepts JSON and returns scoring data. |
| Frontend   | Static HTML/CSS/JS (no build step). Calls FastAPI via `fetch`, renders score cards and keyword badges.                 |
| NLP Assets | `all-MiniLM-L6-v2` for semantic similarity; VADER for sentiment; `language_tool_python` for grammar checks.            |

---

## Scoring Formula

Five rubric criteria are normalized and combined using the official weights (total 100). Each criterion produces feedback plus supporting metrics.

| Criterion (weight)             | How it is scored                                                                                                                                                                                                                                                                 | Max pts |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| **Content & Structure (40)**   | Salutation quality (0–5) + keyword presence (must-have phrases worth 4 pts each, good-to-have worth 2 pts each, scaled to 30) + flow check (0/5) + semantic similarity bonus (0–5). The similarity uses sentence embeddings to capture paraphrases. Score capped at 40.            |
| **Speech Rate (10)**           | `wpm = words / (duration_seconds / 60)`. WPM mapped to rubric bands: 111–140 → 10 pts (ideal), 81–110 or 141–160 → 6 pts, ≥161 or ≤80 → 2 pts.                                                                                                                                  |
| **Language & Grammar (20)**    | Grammar errors per 100 words from LanguageTool mapped to 0–10, plus Type-Token Ratio (vocabulary richness) mapped to 0–10. Sum capped at 20.                                                                                                                                      |
| **Clarity / Filler Words (15)**| Percentage of filler tokens (um, uh, like, etc.): ≤3% → 15 pts, 4–6% → 12 pts, 7–9% → 9 pts, 10–12% → 6 pts, ≥13% → 3 pts.                                                                                                                 |
| **Engagement / Positivity (15)** | VADER compound sentiment score: ≥0.9 → 15, 0.7–0.89 → 12, 0.5–0.69 → 9, 0.3–0.49 → 6, <0.3 → 3.                                                                                                                                           |

The weighted sum of the normalized criterion scores yields the final 0–100 score.

Additional logic:

- Semantic keyword detection augments literal phrase checks using cosine similarity (threshold 0.68) so paraphrased statements still count.
- Flow validation ensures transcripts contain a greeting, identity details, and a closing.
- Output includes filler counts, WPM, sentiment, grammar error rates, and keyword hit maps for quick auditing.

---

## Repository Layout

```
backend/
  main.py          # FastAPI app, schemas, CORS
  scoring.py       # Rubric + NLP scoring logic
  requirements.txt
frontend/
  index.html       # UI
  style.css
  script.js
README.md
DEPLOYMENT_STEPS.md
```

---

## Running Locally (Quick Start)

1. **Clone & enter repo**
   ```bash
   git clone https://github.com/<your-user>/nirmaan-communication-scorer.git
   cd nirmaan-communication-scorer
   ```
2. **Install backend dependencies**
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Start the API**
   ```bash
   PORT=${PORT:-8000}
   uvicorn main:app --reload --host 0.0.0.0 --port $PORT
   ```
   - Health check: `GET http://127.0.0.1:8000/health`
   - Docs: `http://127.0.0.1:8000/docs`
4. **Open the frontend**
   - Option A: open `frontend/index.html` directly in a browser.
   - Option B (recommended): serve it via a static server for consistent CORS.
     ```bash
     cd ../frontend
     python3 -m http.server 5500
     ```
     Then browse to `http://127.0.0.1:5500`.
5. **Paste a transcript, optionally set duration (seconds), click “Score”.**  
   Use the “New Transcript” button to clear the form and score another sample.

---

## API Reference

`POST /score`

Request
```jsonc
{
  "transcript": "string (required)",
  "duration_seconds": 85
}
```

Response
```jsonc
{
  "overall_score": 84.32,
  "words": 301,
  "sentences": 21,
  "duration_seconds": 85,
  "wpm": 212.47,
  "criteria": [
    {
      "id": "content_structure",
      "name": "Content & Structure",
      "points": 37.65,
      "max_points": 40,
      "weight": 40,
      "normalized": 0.941,
      "feedback": "...",
      "details": {
        "must_hits": {"name": true, "age": true, "...": true},
        "good_hits": {"origin": false, "...": true}
      }
    },
    ...
  ],
  "keyword_hits": {
    "must": {"name": true, "age": true, "...": true},
    "good": {"origin": false, "...": true}
  }
}
```

---

## Deployment Notes

- The backend command above reads the `PORT` environment variable so it can run on PaaS providers (Render/Railway/Heroku) that inject dynamic ports.
- The frontend automatically detects the API base:
  - When served from `localhost:5500`, it talks to `http://127.0.0.1:8000`.
  - When frontend and backend are on the same origin, it uses the current origin.
  - To point the UI at a different backend (e.g., production API), define `window.__API_BASE__ = "https://api.example.com";` before loading `script.js`.

---

## Submission Checklist

- ✅ Working backend (FastAPI) and frontend (HTML/CSS/JS)
- ✅ Rubric-driven scoring with rule-based + NLP + weighting
- ✅ README with scoring explanation and run instructions
- ✅ `DEPLOYMENT_STEPS.md` documented for evaluators
- ✅ Optional enhancements: semantic keyword matching, keyword badges, reset button

---

## Future Enhancements

- Export to PDF / downloadable JSON report
- Named-entity extraction to highlight student name in UI
- Hosted deployment on Render/Railway + GitHub Pages with HTTPS
- Fine-tune thresholds per grade level or speech-length cohort

Contributions and suggestions are welcome!

