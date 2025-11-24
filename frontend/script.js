function deriveApiBase() {
  if (window.__API_BASE__) {
    return window.__API_BASE__;
  }
  const origin = window.location.origin;
  if (origin.includes("127.0.0.1:5500") || origin.includes("localhost:5500")) {
    return origin.replace("5500", "8000");
  }
  return origin;
}

const API_BASE = deriveApiBase();

const transcriptEl = document.getElementById("transcript");
const durationEl = document.getElementById("duration");
const scoreBtn = document.getElementById("scoreBtn");
const overallCard = document.getElementById("overallCard");
const keywordCard = document.getElementById("keywordCard");
const criteriaCard = document.getElementById("criteriaCard");
const resetBtn = document.getElementById("resetBtn");

scoreBtn.addEventListener("click", async () => {
  const transcript = transcriptEl.value.trim();
  const durationValue = durationEl.value.trim();

  if (!transcript) {
    alert("Please paste the transcript first.");
    return;
  }

  let durationSeconds = null;
  if (durationValue !== "") {
    const num = Number(durationValue);
    if (!Number.isFinite(num) || num < 0) {
      alert("Duration must be a non-negative number.");
      return;
    }
    durationSeconds = num;
  }

  scoreBtn.disabled = true;
  scoreBtn.textContent = "Scoring...";

  try {
    const res = await fetch(`${API_BASE}/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transcript,
        duration_seconds: durationSeconds,
      }),
    });

    if (!res.ok) {
      throw new Error("API error " + res.status);
    }

    const data = await res.json();
    renderOverall(data);
    renderKeywordBadges(data.keyword_hits);
    renderCriteria(data.criteria);
  } catch (err) {
    console.error(err);
    alert("Something went wrong while scoring. Check console for details.");
  } finally {
    scoreBtn.disabled = false;
    scoreBtn.textContent = "Score";
  }
});

resetBtn.addEventListener("click", () => {
  transcriptEl.value = "";
  durationEl.value = "";
  clearOutputs();
  transcriptEl.focus();
});

function renderOverall(data) {
  overallCard.classList.remove("hidden");
  overallCard.innerHTML = `
    <h2>Overall Score</h2>
    <div class="score-main">${data.overall_score}</div>
    <div class="meta">
      Words: <strong>${data.words}</strong> &nbsp;|&nbsp;
      Sentences: <strong>${data.sentences}</strong>
      ${
        data.wpm
          ? `&nbsp;|&nbsp; Estimated WPM: <strong>${data.wpm}</strong>`
          : ""
      }
    </div>
    <p class="meta">
      This score combines rule-based checks (keywords, grammar, filler words),
      NLP-based semantic similarity & sentiment, and rubric weights.
    </p>
  `;
}

function clearOutputs() {
  overallCard.classList.add("hidden");
  keywordCard.classList.add("hidden");
  criteriaCard.classList.add("hidden");
  overallCard.innerHTML = "";
  keywordCard.innerHTML = "";
  criteriaCard.innerHTML = "";
}

function formatLabel(label) {
  return label
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderKeywordBadges(keywordHits) {
  if (!keywordHits) {
    keywordCard.classList.add("hidden");
    return;
  }

  const renderGroup = (title, hits) => {
    if (!hits) return "";
    const badges = Object.entries(hits)
      .map(
        ([label, hit]) => `
        <span class="badge ${hit ? "badge-hit" : "badge-miss"}">
          ${hit ? "✓" : "✕"} ${formatLabel(label)}
        </span>
      `
      )
      .join("");
    return `
      <div class="keyword-group">
        <h3>${title}</h3>
        <div class="badge-grid">
          ${badges || "<span class='meta'>No keywords configured.</span>"}
        </div>
      </div>
    `;
  };

  keywordCard.classList.remove("hidden");
  keywordCard.innerHTML = `
    <h2>Keyword Checklist</h2>
    <p class="meta">Quick visual rubric check for required details.</p>
    ${renderGroup("Must-have items", keywordHits.must)}
    ${renderGroup("Good-to-have items", keywordHits.good)}
  `;
}

function renderCriteria(criteria) {
  criteriaCard.classList.remove("hidden");

  const rows = criteria
    .map((c) => {
      const ratio = (c.normalized * 100).toFixed(1);
      let extra = "";

      if (c.details) {
        const d = c.details;
        const detailParts = [];
        if (typeof d.wpm !== "undefined") {
          detailParts.push(`WPM: ${d.wpm}`);
        }
        if (typeof d.filler_count !== "undefined") {
          detailParts.push(
            `Filler: ${d.filler_count} (${d.filler_rate}% of words)`
          );
        }
        if (typeof d.sentiment_compound !== "undefined") {
          detailParts.push(`Sentiment: ${d.sentiment_compound}`);
        }
        if (d.must_hits) {
          const hits = Object.entries(d.must_hits)
            .map(([label, hit]) => `${formatLabel(label)}: ${hit ? "✓" : "✕"}`)
            .join(", ");
          detailParts.push(`Must-have: ${hits}`);
        }
        if (d.good_hits) {
          const hits = Object.entries(d.good_hits)
            .map(([label, hit]) => `${formatLabel(label)}: ${hit ? "✓" : "✕"}`)
            .join(", ");
          detailParts.push(`Good-to-have: ${hits}`);
        }
        if (detailParts.length > 0) {
          extra = `<div class="meta">${detailParts.join(" | ")}</div>`;
        }
      }

      return `
        <tr>
          <td>${c.name}</td>
          <td>${c.points} / ${c.max_points}</td>
          <td>${c.weight}</td>
          <td>${ratio}%</td>
          <td>
            <div class="feedback">${c.feedback}</div>
            ${extra}
          </td>
        </tr>
      `;
    })
    .join("");

  criteriaCard.innerHTML = `
    <h2>Per-criterion Feedback</h2>
    <table class="criteria-table">
      <thead>
        <tr>
          <th>Criterion</th>
          <th>Score</th>
          <th>Weight (Rubric)</th>
          <th>Normalized</th>
          <th>Feedback & Details</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}
