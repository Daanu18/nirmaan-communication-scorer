# Local Deployment Guide – Nirmaan AI Communication Scorer

This document expands the run instructions from the README into a step-by-step checklist for evaluators. The process takes ~5 minutes on macOS/Linux with Python 3.9+ installed.

---

## 1. Clone the repository

```bash
git clone https://github.com/<your-user>/nirmaan-communication-scorer.git
cd nirmaan-communication-scorer
```

> Replace `<your-user>` with your GitHub handle if the project lives under your account.

---

## 2. Prepare the backend environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate           # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The first install may take a few minutes because PyTorch + transformers are downloaded.
- If you see `TOKENIZERS_PARALLELISM` warnings later, they are safe to ignore.

---

## 3. Run the FastAPI server

```bash
PORT=${PORT:-8000}
uvicorn main:app --reload --host 0.0.0.0 --port $PORT
```

Expected output:
```
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Useful endpoints:
- `http://127.0.0.1:8000/health` → simple `{"status": "ok"}` check
- `http://127.0.0.1:8000/docs` → interactive Swagger UI to call `/score`

Keep this terminal open while testing.

---

## 4. Serve the frontend

Open another terminal window/tab:

```bash
cd /path/to/nirmaan-communication-scorer/frontend
python3 -m http.server 5500
```

Navigate to `http://127.0.0.1:5500` in the browser.  
Alternatively, you can double-click `frontend/index.html`, but running a static server avoids CORS caching issues.

---

## 5. Test end-to-end

1. Browse to the frontend.
2. Paste a transcript into the textarea. (Use the sample from the case-study Excel to verify expected scores.)
3. Optionally fill “Duration in seconds” so WPM is calculated accurately.
4. Click **Score**. Watch the backend terminal for request logs and ensure the UI shows the overall/criterion cards plus the keyword badge checklist.
5. Click **New Transcript** to clear the form and test another sample.

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `OPTIONS /score 405` in backend logs | Ensure FastAPI server was restarted after installing dependencies (CORS middleware is already configured). |
| Browser cannot reach API | Confirm the backend is listening on the expected port (via the `PORT` env var) and, if hosting on a different domain, set `window.__API_BASE__ = "https://your-api-url";` before loading `script.js`. |
| `PermissionError` when importing torch | Re-run the `pip install` step outside of sandboxed terminals (or use `required_permissions: ['all']` if in Cursor). |
| LanguageTool download failure | Ensure you have network access the first time; the library caches data locally afterwards. |

---

## 7. Stopping services

When done:

- Press `Ctrl+C` in the frontend static-server terminal.
- Press `Ctrl+C` in the `uvicorn` terminal to stop FastAPI.
- Deactivate the virtual environment with `deactivate` (optional).

---

Following the above steps ensures reviewers can reproduce the scoring experience exactly as required in the case study. Let me know if you need deployment steps for a hosted environment (Render/Railway/EC2).***

